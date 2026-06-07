"""
Detección de comunidades sobre red_monto y estadísticas por comunidad.

  Entrada : output/graphs/red_monto.gexf
  Salidas : output/graphs/red_monto_comunidades.gexf   (con atributo 'comunidad')
            output/tables/comunidades_stats.csv
            output/figures/histogramas_comunidades.png
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities

from . import config

# atributos numéricos de escala presentes en red_monto
FEATURES_NUM = ["monto_total", "monto_max", "log_monto", "n_periodos"]
TOP_N_COMUNIDADES = 4

STATS_CSV = config.TABLES_DIR / "comunidades_stats.csv"
HISTOGRAMAS_PNG = config.FIGURES_DIR / "histogramas_comunidades.png"


def detectar_comunidades(G):
    print("Detectando comunidades (greedy modularity) ...")
    comunidades = greedy_modularity_communities(G, weight="weight")
    comunidades = sorted(comunidades, key=len, reverse=True)
    for cid, nodo_set in enumerate(comunidades):
        for n in nodo_set:
            G.nodes[n]["comunidad"] = cid
    print(f"Comunidades detectadas: {len(comunidades)}")
    print("Tamaños top-10:", [len(c) for c in comunidades[:10]])
    return G, comunidades


def nodos_a_dataframe(G):
    filas = [{"cuit": n, **attrs} for n, attrs in G.nodes(data=True)]
    df = pd.DataFrame(filas).set_index("cuit")
    for c in FEATURES_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def stats_por_comunidad(df, comunidades, csv_out=STATS_CSV):
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
    stats.to_csv(str(csv_out))
    print(f"CSV: {csv_out}")
    return df_top


def plotear_histogramas(df_top, png_out=HISTOGRAMAS_PNG):
    features_presentes = [f for f in FEATURES_NUM if f in df_top.columns]
    if not features_presentes or df_top.empty:
        print("Sin features para histogramas.")
        return

    n_features = len(features_presentes)
    n_comunidades = df_top["comunidad"].nunique()
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        n_features, 1, figsize=(12, 3 * n_features), constrained_layout=True
    )
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_presentes):
        for cid in sorted(df_top["comunidad"].unique()):
            vals = df_top.loc[df_top["comunidad"] == cid, feat].dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=cmap(cid % 10),
                    label=f"C{cid} (n={len(vals):,})", density=True)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("densidad")
        ax.legend(fontsize=8, ncol=n_comunidades)

    fig.suptitle("Distribución de features por comunidad (top N)", fontsize=13)
    fig.savefig(str(png_out), dpi=150)
    print(f"PNG: {png_out}")
    plt.close(fig)


def main():
    config.asegurar_dirs(config.GRAPHS_DIR, config.TABLES_DIR, config.FIGURES_DIR)

    if not config.RED_MONTO_GEXF.exists():
        print(f"WARN: {config.RED_MONTO_GEXF} no existe — corré build_red_monto primero.")
        return

    print(f"Cargando {config.RED_MONTO_GEXF.name} ...")
    G = nx.read_gexf(str(config.RED_MONTO_GEXF))
    print(f"Nodos: {G.number_of_nodes():,} | Aristas: {G.number_of_edges():,}")

    if G.number_of_edges() == 0:
        print("WARN: el grafo no tiene aristas — sin detección de comunidades.")
        return

    G, comunidades = detectar_comunidades(G)
    nx.write_gexf(G, str(config.RED_MONTO_COMUNIDADES_GEXF))
    print(f"GEXF enriquecido: {config.RED_MONTO_COMUNIDADES_GEXF}")

    df = nodos_a_dataframe(G)
    df_top = stats_por_comunidad(df, comunidades)
    plotear_histogramas(df_top)


if __name__ == "__main__":
    main()
