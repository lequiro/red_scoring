#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación: cruza comunidades de escala (red_monto) con la etiqueta de
comportamiento por CUIT (calculada en build_features.py, no del grafo red_morosos).
  Pregunta: ¿los deudores grandes son más o menos morosos que los chicos?
  Salida  : output/tables/cruce_comunidad_comportamiento.csv
            output/figures/cruce_heatmap.png

Correr: python3 validacion_cruce.py
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

AQUI         = os.path.dirname(os.path.abspath(__file__))
# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR      = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR   = os.path.join(OUT_DIR, "graphs")
TABLES_DIR   = os.path.join(OUT_DIR, "tables")
FIGURES_DIR  = os.path.join(OUT_DIR, "figures")
GEXF_MONTO   = os.path.join(GRAPHS_DIR, "red_monto_comunidades.gexf")
FEATURES_CSV = os.path.join(TABLES_DIR, "features.csv")
CSV_OUT      = os.path.join(TABLES_DIR, "cruce_comunidad_comportamiento.csv")
PNG_OUT      = os.path.join(FIGURES_DIR, "cruce_heatmap.png")

TOP_N = 4   # solo las N comunidades más grandes


# ── carga ──────────────────────────────────────────────────────────────────────

def gexf_a_df(path, attrs):
    """Extrae atributos de nodos de un GEXF como DataFrame."""
    G = nx.read_gexf(path)
    filas = []
    for n, data in G.nodes(data=True):
        fila = {"cuit": n}
        for a in attrs:
            fila[a] = data.get(a, None)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("cuit")


# ── análisis ───────────────────────────────────────────────────────────────────

def cruzar(df_monto, df_comp):
    df = df_monto.join(df_comp, how="outer")
    print(f"CUITs en red_monto      : {df_monto.index.nunique():,}")
    print(f"CUITs con comportamiento: {df_comp.index.nunique():,}")
    print(f"Solo en red_monto       : {df['comportamiento'].isna().sum():,}")
    print(f"Solo con comportamiento : {df['comunidad'].isna().sum():,}")
    print(f"En ambos                : {df.dropna(subset=['comportamiento','comunidad']).__len__():,}")
    df = df.dropna(subset=["comportamiento", "comunidad"])
    df["comunidad"] = df["comunidad"].astype(int)
    return df


def _labels_por_escala(df, comunidades):
    """
    Etiqueta cada comunidad por su escala REAL (log_monto medio), no por su id.

    El id de comunidad lo asigna analisis_comunidades.py por TAMAÑO (0 = la más
    grande en nº de nodos), que no tiene relación con la escala económica. Acá
    rankeamos por log_monto medio: esc1 = menor escala. Incluye la media para
    que el lector vea el valor, sin rótulos engañosos del tipo "grande/chico".
    """
    media = df.groupby("comunidad")["log_monto"].mean()
    orden = media.loc[list(comunidades)].sort_values()
    return {
        cid: f"C{int(cid)}·esc{rank} (m={m:.1f})"
        for rank, (cid, m) in enumerate(orden.items(), start=1)
    }


def tabla_cruce(df):
    """Tabla de contingencia: comunidad (filas) x comportamiento (columnas)."""
    top = sorted(df["comunidad"].value_counts().index[:TOP_N])
    df_top = df[df["comunidad"].isin(top)].copy()
    labels = _labels_por_escala(df_top, top)
    df_top["comunidad_label"] = df_top["comunidad"].map(labels)

    tabla = pd.crosstab(
        df_top["comunidad_label"],
        df_top["comportamiento"],
        margins=True,
        margins_name="Total",
    )
    tabla_pct = pd.crosstab(
        df_top["comunidad_label"],
        df_top["comportamiento"],
        normalize="index",
    ).round(3) * 100

    print("\n── Conteos ──")
    print(tabla.to_string())
    print("\n── % por comunidad (fila) ──")
    print(tabla_pct.to_string())

    tabla.to_csv(CSV_OUT)
    print(f"\nCSV: {CSV_OUT}")
    return tabla_pct


def plotear_heatmap(tabla_pct):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(tabla_pct.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(tabla_pct.columns)))
    ax.set_xticklabels(tabla_pct.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(tabla_pct.index)))
    ax.set_yticklabels(tabla_pct.index)

    for i in range(len(tabla_pct.index)):
        for j in range(len(tabla_pct.columns)):
            ax.text(j, i, f"{tabla_pct.values[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="black" if tabla_pct.values[i, j] < 50 else "white")

    plt.colorbar(im, ax=ax, label="% dentro de comunidad")
    ax.set_title("Comportamiento por comunidad de escala (%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=150)
    print(f"PNG: {PNG_OUT}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_monto = gexf_a_df(GEXF_MONTO, ["comunidad", "log_monto", "monto_total"])
    df_comp  = pd.read_csv(FEATURES_CSV, index_col=0)[["comportamiento"]]

    df = cruzar(df_monto, df_comp)
    tabla_pct = tabla_cruce(df)
    plotear_heatmap(tabla_pct)


if __name__ == "__main__":
    main()
