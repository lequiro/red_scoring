#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detección de comunidades sobre red_monto.gexf (red principal del proyecto).

  Salidas:
    output/graphs/red_monto_comunidades.gexf
    output/tables/comunidades_stats.csv
    output/figures/histogramas_comunidades.png

Correr: python3 analisis_comunidades.py
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities

AQUI = os.path.dirname(os.path.abspath(__file__))
# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

# ── paths por grafo ────────────────────────────────────────────────────────────

# red_morosos quedó fuera del pipeline: solo se procesa red_monto.
GRAFOS = {
    "monto": {
        "gexf_in":  os.path.join(GRAPHS_DIR, "red_monto.gexf"),
        "gexf_out": os.path.join(GRAPHS_DIR, "red_monto_comunidades.gexf"),
        "csv_out":  os.path.join(TABLES_DIR, "comunidades_stats.csv"),
        "png_out":  os.path.join(FIGURES_DIR, "histogramas_comunidades.png"),
    },
}

# Features numéricas a analizar — el script usa solo los que existan en el GEXF cargado
FEATURES_NUM = [
    "peor_situacion", "situacion_final", "n_periodos_mora",
    "racha_max_mora", "tendencia", "n_transiciones", "volatilidad",
    "monto_total", "monto_max", "log_monto", "n_periodos",
]
TOP_N_COMUNIDADES = 4


# ── 1. carga y detección ───────────────────────────────────────────────────────

def cargar_grafo(gexf_in):
    print(f"Cargando {gexf_in} ...")
    G = nx.read_gexf(gexf_in)
    print(f"Nodos: {G.number_of_nodes():,} | Aristas: {G.number_of_edges():,}")
    return G


def detectar_comunidades(G):
    print("Detectando comunidades (greedy modularity) ...")
    comunidades = greedy_modularity_communities(G, weight="weight")
    comunidades = sorted(comunidades, key=len, reverse=True)
    for nodo_set, cid in zip(comunidades, range(len(comunidades))):
        for n in nodo_set:
            G.nodes[n]["comunidad"] = cid
    print(f"Comunidades detectadas: {len(comunidades)}")
    print("Tamaños top-10:", [len(c) for c in comunidades[:10]])
    return G, comunidades


# ── 2. tabla de features ───────────────────────────────────────────────────────

def nodos_a_dataframe(G):
    filas = []
    for n, attrs in G.nodes(data=True):
        fila = {"cuit": n}
        fila.update(attrs)
        filas.append(fila)
    df = pd.DataFrame(filas).set_index("cuit")
    for c in FEATURES_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 3. estadísticas por comunidad ─────────────────────────────────────────────

def stats_por_comunidad(df, comunidades, csv_out):
    top_n = min(TOP_N_COMUNIDADES, len(comunidades))
    df_top = df[df["comunidad"].isin(range(top_n))]
    features_presentes = [f for f in FEATURES_NUM if f in df_top.columns]
    if not features_presentes:
        print("Sin features numéricas para estadísticas.")
        return df_top

    stats = (
        df_top.groupby("comunidad")[features_presentes]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    print("\n── Estadísticas por comunidad ──")
    print(stats.to_string())
    stats.to_csv(csv_out)
    print(f"CSV: {csv_out}")
    return df_top


# ── 4. histogramas ─────────────────────────────────────────────────────────────

def plotear_histogramas(df_top, png_out):
    features_presentes = [f for f in FEATURES_NUM if f in df_top.columns]
    if not features_presentes or df_top.empty:
        print("Sin features para histogramas.")
        return

    n_features = len(features_presentes)
    n_comunidades = df_top["comunidad"].nunique()
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(12, 3 * n_features),
        constrained_layout=True,
    )
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_presentes):
        for cid in sorted(df_top["comunidad"].unique()):
            vals = df_top.loc[df_top["comunidad"] == cid, feat].dropna()
            ax.hist(
                vals, bins=30, alpha=0.5,
                color=cmap(cid % 10),
                label=f"C{cid} (n={len(vals):,})",
                density=True,
            )
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("densidad")
        ax.legend(fontsize=8, ncol=n_comunidades)

    fig.suptitle("Distribución de features por comunidad (top N)", fontsize=13)
    fig.savefig(png_out, dpi=150)
    print(f"PNG: {png_out}")
    plt.close(fig)


# ── pipeline por grafo ─────────────────────────────────────────────────────────

def procesar_grafo(nombre, paths):
    print(f"\n{'='*60}")
    print(f"Procesando grafo: {nombre.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(paths["gexf_in"]):
        print(f"WARN: {paths['gexf_in']} no existe — saltando {nombre}.")
        return

    G = cargar_grafo(paths["gexf_in"])

    if G.number_of_edges() == 0:
        print(f"WARN: grafo {nombre} no tiene aristas — saltando detección de comunidades.")
        return

    G, comunidades = detectar_comunidades(G)
    nx.write_gexf(G, paths["gexf_out"])
    print(f"GEXF enriquecido: {paths['gexf_out']}")

    df = nodos_a_dataframe(G)
    df_top = stats_por_comunidad(df, comunidades, paths["csv_out"])
    plotear_histogramas(df_top, paths["png_out"])


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    for d in (GRAPHS_DIR, TABLES_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)
    for nombre, paths in GRAFOS.items():
        procesar_grafo(nombre, paths)


if __name__ == "__main__":
    main()
