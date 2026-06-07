"""
Validaciones del modelo de escala (red_monto) contra la etiqueta de
comportamiento por CUIT (calculada en behavior.py, no del grafo red_morosos).

  main_cruce      : cruza comunidad de escala x comportamiento (tabla + heatmap).
  main_entidades  : controla si el patrón está sesgado por qué bancos operan en
                    cada segmento (top entidades + tasa de mora por entidad).

Salidas en output/tables/ y output/figures/.
"""

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .io import gexf_nodos_a_df, cargar_csv_basico

TOP_N_COMUNIDADES = 4
TOP_N_ENTIDADES = 5

# salidas
CRUCE_CSV = config.TABLES_DIR / "cruce_comunidad_comportamiento.csv"
CRUCE_PNG = config.FIGURES_DIR / "cruce_heatmap.png"
ENT_TOP_CSV = config.TABLES_DIR / "validacion_entidades_top.csv"
ENT_MORA_CSV = config.TABLES_DIR / "validacion_mora_por_entidad.csv"
ENT_PNG = config.FIGURES_DIR / "validacion_entidades_heatmap.png"


def labels_por_escala(df, comunidades=None) -> dict:
    """
    Etiqueta cada comunidad por su escala REAL (log_monto medio), no por su id.

    El id lo asigna communities.py por TAMAÑO (0 = la más grande en nº de nodos),
    sin relación con la escala económica. Acá rankeamos por log_monto medio
    (esc1 = menor escala) y mostramos la media, sin rótulos engañosos.
    """
    media = df.groupby("comunidad")["log_monto"].mean()
    if comunidades is not None:
        media = media.loc[list(comunidades)]
    media = media.sort_values()
    return {
        cid: f"C{int(cid)}·esc{rank} (m={m:.1f})"
        for rank, (cid, m) in enumerate(media.items(), start=1)
    }


# ── cruce comunidad x comportamiento ─────────────────────────────────────────

def cruzar(df_monto, df_comp):
    df = df_monto.join(df_comp, how="outer")
    print(f"CUITs en red_monto      : {df_monto.index.nunique():,}")
    print(f"CUITs con comportamiento: {df_comp.index.nunique():,}")
    print(f"Solo en red_monto       : {df['comportamiento'].isna().sum():,}")
    print(f"Solo con comportamiento : {df['comunidad'].isna().sum():,}")
    print(f"En ambos                : {len(df.dropna(subset=['comportamiento','comunidad'])):,}")
    df = df.dropna(subset=["comportamiento", "comunidad"])
    df["comunidad"] = df["comunidad"].astype(int)
    return df


def tabla_cruce(df):
    """Tabla de contingencia: comunidad (filas) x comportamiento (columnas)."""
    top = sorted(df["comunidad"].value_counts().index[:TOP_N_COMUNIDADES])
    df_top = df[df["comunidad"].isin(top)].copy()
    labels = labels_por_escala(df_top, top)
    df_top["comunidad_label"] = df_top["comunidad"].map(labels)

    tabla = pd.crosstab(df_top["comunidad_label"], df_top["comportamiento"],
                        margins=True, margins_name="Total")
    tabla_pct = pd.crosstab(df_top["comunidad_label"], df_top["comportamiento"],
                            normalize="index").round(3) * 100

    print("\n── Conteos ──")
    print(tabla.to_string())
    print("\n── % por comunidad (fila) ──")
    print(tabla_pct.to_string())

    tabla.to_csv(str(CRUCE_CSV))
    print(f"\nCSV: {CRUCE_CSV}")
    return tabla_pct


def plotear_heatmap_cruce(tabla_pct):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(tabla_pct.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(tabla_pct.columns)))
    ax.set_xticklabels(tabla_pct.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(tabla_pct.index)))
    ax.set_yticklabels(tabla_pct.index)
    for i in range(len(tabla_pct.index)):
        for j in range(len(tabla_pct.columns)):
            ax.text(j, i, f"{tabla_pct.values[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if tabla_pct.values[i, j] < 50 else "white")
    plt.colorbar(im, ax=ax, label="% dentro de comunidad")
    ax.set_title("Comportamiento por comunidad de escala (%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(CRUCE_PNG), dpi=150)
    print(f"PNG: {CRUCE_PNG}")
    plt.close(fig)


def _leer_comportamiento():
    """features.csv con el cuit forzado a str (si no, pandas lo infiere int y el
    join contra los ids de nodo del GEXF —strings— no matchea nada)."""
    df = pd.read_csv(str(config.FEATURES_CSV), index_col=0)[["comportamiento"]]
    df.index = df.index.astype(str)
    return df


def main_cruce():
    config.asegurar_dirs(config.TABLES_DIR, config.FIGURES_DIR)
    df_monto = gexf_nodos_a_df(config.RED_MONTO_COMUNIDADES_GEXF,
                               ["comunidad", "log_monto", "monto_total"])
    df_comp = _leer_comportamiento()
    df = cruzar(df_monto, df_comp)
    tabla_pct = tabla_cruce(df)
    plotear_heatmap_cruce(tabla_pct)


# ── validación por entidad ───────────────────────────────────────────────────

def construir_base(df_csv, df_monto, df_comp):
    """Une CSV con comunidad (red_monto) y comportamiento (etiqueta por CUIT)."""
    meta = df_monto[["comunidad", "log_monto"]].join(df_comp[["comportamiento"]], how="inner")
    meta = meta.dropna(subset=["comunidad", "comportamiento"])
    meta["comunidad"] = meta["comunidad"].astype(int)
    meta["log_monto"] = pd.to_numeric(meta["log_monto"], errors="coerce")
    meta = meta[meta["comunidad"] < TOP_N_COMUNIDADES]
    meta["comunidad_label"] = meta["comunidad"].map(labels_por_escala(meta))
    meta = meta.drop(columns=["log_monto"])

    base = df_csv.join(meta, on="cuit", how="inner")
    print(f"Filas con comunidad asignada: {len(base):,}")
    return base


def top_entidades_por_comunidad(base):
    """Top-N entidades por frecuencia de aparición en cada comunidad."""
    freq = (base.groupby(["comunidad_label", "entidad"]).size()
            .reset_index(name="n_registros"))
    top = (freq.sort_values("n_registros", ascending=False)
           .groupby("comunidad_label").head(TOP_N_ENTIDADES)
           .sort_values(["comunidad_label", "n_registros"], ascending=[True, False]))
    print("\n── Top entidades por comunidad ──")
    print(top.to_string(index=False))
    top.to_csv(str(ENT_TOP_CSV), index=False)
    return top


def mora_por_entidad_comunidad(base):
    """Tasa de mora (% filas con situacion >= 2) por entidad y comunidad."""
    base = base[base["monto"] > 0].copy()
    base["en_mora"] = (base["situacion"] >= 2).astype(int)

    agg = base.groupby(["comunidad_label", "entidad"]).agg(
        n_registros=("en_mora", "count"),
        n_mora=("en_mora", "sum"),
    ).reset_index()
    agg["tasa_mora_pct"] = (agg["n_mora"] / agg["n_registros"] * 100).round(1)
    agg = agg[agg["n_registros"] >= 20]  # estabilidad de la tasa

    top_mora = (agg.sort_values("tasa_mora_pct", ascending=False)
                .groupby("comunidad_label").head(TOP_N_ENTIDADES)
                .sort_values(["comunidad_label", "tasa_mora_pct"], ascending=[True, False]))
    print("\n── Tasa de mora por entidad (top por comunidad, min 20 registros) ──")
    print(top_mora.to_string(index=False))
    top_mora.to_csv(str(ENT_MORA_CSV), index=False)
    return agg


def plotear_heatmap_entidades(agg):
    """Heatmap: entidad (filas) x comunidad (columnas) — tasa de mora %."""
    presencia = agg.groupby("entidad")["comunidad_label"].nunique()
    ents_transversales = presencia[presencia >= 2].index

    pivot = (agg[agg["entidad"].isin(ents_transversales)]
             .pivot_table(index="entidad", columns="comunidad_label",
                          values="tasa_mora_pct", aggfunc="mean")
             .dropna(thresh=2))

    if pivot.empty:
        print("Sin entidades transversales suficientes para el heatmap.")
        return

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(20)
    vmax = pivot.values[~pd.isna(pivot.values)].max()

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if val > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="tasa mora %")
    ax.set_title("Tasa de mora por entidad y comunidad de escala", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(ENT_PNG), dpi=150)
    print(f"\nPNG: {ENT_PNG}")
    plt.close(fig)


def main_entidades():
    config.asegurar_dirs(config.TABLES_DIR, config.FIGURES_DIR)
    df_csv = cargar_csv_basico(config.CSV_MERGED)
    df_monto = gexf_nodos_a_df(config.RED_MONTO_COMUNIDADES_GEXF, ["comunidad", "log_monto"])
    df_comp = _leer_comportamiento()

    base = construir_base(df_csv, df_monto, df_comp)
    top_entidades_por_comunidad(base)
    agg = mora_por_entidad_comunidad(base)
    plotear_heatmap_entidades(agg)
