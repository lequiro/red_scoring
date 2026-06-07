#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación por entidad financiera.
  Pregunta: ¿el patrón de comportamiento por comunidad de escala es genuino,
            o está sesgado por qué bancos operan en cada segmento?

  Para cada (comunidad, comportamiento):
    - Top-5 entidades más frecuentes
    - Tasa de mora por entidad dentro de cada comunidad

  Salida: output/validacion_entidades_top.csv
          output/validacion_mora_por_entidad.csv
          output/validacion_entidades_heatmap.png

Correr: python3 validacion_entidades.py
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

AQUI         = os.path.dirname(os.path.abspath(__file__))
CSV          = os.path.normpath(os.path.join(AQUI, "..", "data", "bcra_merged.csv"))
# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR      = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR   = os.path.join(OUT_DIR, "graphs")
TABLES_DIR   = os.path.join(OUT_DIR, "tables")
FIGURES_DIR  = os.path.join(OUT_DIR, "figures")
GEXF_MONTO   = os.path.join(GRAPHS_DIR, "red_monto_comunidades.gexf")
FEATURES_CSV = os.path.join(TABLES_DIR, "features.csv")
OUT_TOP      = os.path.join(TABLES_DIR, "validacion_entidades_top.csv")
OUT_MORA     = os.path.join(TABLES_DIR, "validacion_mora_por_entidad.csv")
OUT_PNG      = os.path.join(FIGURES_DIR, "validacion_entidades_heatmap.png")

TOP_N_COMUNIDADES = 4
TOP_N_ENTIDADES   = 5


# ── carga ──────────────────────────────────────────────────────────────────────

def cargar_atributos_nodos(path, attrs):
    G = nx.read_gexf(path)
    return pd.DataFrame(
        [{"cuit": n, **{a: d.get(a) for a in attrs}} for n, d in G.nodes(data=True)]
    ).set_index("cuit")


def cargar_csv():
    df = pd.read_csv(CSV, dtype={"cuit": str})
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df["situacion"] = pd.to_numeric(df["situacion"], errors="coerce")
    return df[df["entidad"].notna()]


# ── análisis ───────────────────────────────────────────────────────────────────

def _labels_por_escala(meta):
    """
    Etiqueta cada comunidad por su escala REAL (log_monto medio), no por su id.

    El id lo asigna analisis_comunidades.py por TAMAÑO (0 = la más grande en nº
    de nodos), sin relación con la escala económica. Acá rankeamos por log_monto
    medio (esc1 = menor escala) y mostramos la media, sin rótulos engañosos.
    """
    media = meta.groupby("comunidad")["log_monto"].mean().sort_values()
    return {
        cid: f"C{int(cid)}·esc{rank} (m={m:.1f})"
        for rank, (cid, m) in enumerate(media.items(), start=1)
    }


def construir_base(df_csv, df_monto, df_comp):
    """Une CSV con comunidad (red_monto) y comportamiento (etiqueta por CUIT)."""
    meta = df_monto[["comunidad", "log_monto"]].join(
        df_comp[["comportamiento"]], how="inner"
    )
    meta = meta.dropna(subset=["comunidad", "comportamiento"])
    meta["comunidad"] = meta["comunidad"].astype(int)
    meta["log_monto"] = pd.to_numeric(meta["log_monto"], errors="coerce")
    meta = meta[meta["comunidad"] < TOP_N_COMUNIDADES]
    labels = _labels_por_escala(meta)
    meta["comunidad_label"] = meta["comunidad"].map(labels)
    meta = meta.drop(columns=["log_monto"])

    # une con el CSV para tener entidad por fila
    base = df_csv.join(meta, on="cuit", how="inner")
    print(f"Filas con comunidad asignada: {len(base):,}")
    return base


def top_entidades_por_comunidad(base):
    """Top-N entidades por frecuencia de aparición en cada comunidad."""
    freq = (
        base.groupby(["comunidad_label", "entidad"])
        .size()
        .reset_index(name="n_registros")
    )
    top = (
        freq.sort_values("n_registros", ascending=False)
        .groupby("comunidad_label")
        .head(TOP_N_ENTIDADES)
        .sort_values(["comunidad_label", "n_registros"], ascending=[True, False])
    )
    print("\n── Top entidades por comunidad ──")
    print(top.to_string(index=False))
    top.to_csv(OUT_TOP, index=False)
    return top


def mora_por_entidad_comunidad(base):
    """
    Tasa de mora (% filas con situacion >= 2) por entidad y comunidad.
    Muestra si los bancos con más crónicos están concentrados en C3.
    """
    base = base[base["monto"] > 0].copy()
    base["en_mora"] = (base["situacion"] >= 2).astype(int)

    agg = base.groupby(["comunidad_label", "entidad"]).agg(
        n_registros=("en_mora", "count"),
        n_mora=("en_mora", "sum"),
    ).reset_index()
    agg["tasa_mora_pct"] = (agg["n_mora"] / agg["n_registros"] * 100).round(1)

    # solo entidades con >= 20 registros para que la tasa sea estable
    agg = agg[agg["n_registros"] >= 20]

    top_mora = (
        agg.sort_values("tasa_mora_pct", ascending=False)
        .groupby("comunidad_label")
        .head(TOP_N_ENTIDADES)
        .sort_values(["comunidad_label", "tasa_mora_pct"], ascending=[True, False])
    )
    print("\n── Tasa de mora por entidad (top por comunidad, min 20 registros) ──")
    print(top_mora.to_string(index=False))
    top_mora.to_csv(OUT_MORA, index=False)
    return agg


def plotear_heatmap_entidades(agg):
    """Heatmap: entidad (filas) x comunidad (columnas) — tasa de mora %."""
    # entidades con registros en >= 2 comunidades (las más transversales)
    presencia = agg.groupby("entidad")["comunidad_label"].nunique()
    ents_transversales = presencia[presencia >= 2].index

    pivot = (
        agg[agg["entidad"].isin(ents_transversales)]
        .pivot_table(index="entidad", columns="comunidad_label",
                     values="tasa_mora_pct", aggfunc="mean")
        .dropna(thresh=2)   # al menos 2 comunidades con dato
    )

    if pivot.empty:
        print("Sin entidades transversales suficientes para el heatmap.")
        return

    # ordenar por tasa promedio
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(20)  # top 20 entidades

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=pivot.values[~pd.isna(pivot.values)].max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7,
                        color="white" if val > pivot.values[~pd.isna(pivot.values)].max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="tasa mora %")
    ax.set_title("Tasa de mora por entidad y comunidad de escala", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nPNG: {OUT_PNG}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_csv   = cargar_csv()
    df_monto = cargar_atributos_nodos(GEXF_MONTO, ["comunidad", "log_monto"])
    df_comp  = pd.read_csv(FEATURES_CSV, index_col=0)[["comportamiento"]]

    base = construir_base(df_csv, df_monto, df_comp)
    top_entidades_por_comunidad(base)
    agg = mora_por_entidad_comunidad(base)
    plotear_heatmap_entidades(agg)


if __name__ == "__main__":
    main()
