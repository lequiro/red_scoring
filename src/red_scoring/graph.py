"""
Construcción de la red de escala económica (red_monto) — red principal.

  Nodos      : CUITs con monto positivo (los que tienen escala medible).
  Condición 1: comparten >= 1 entidad financiera
  Condición 2: |log_monto_i - log_monto_j| < DELTA_MAX
  Peso arista: 1 - |diff| / DELTA_MAX  →  (0, 1], más alto = más similares
  Salida     : output/graphs/red_monto.gexf
"""

from collections import defaultdict

import numpy as np
import networkx as nx
from scipy import sparse

from . import config
from .io import cargar_datos, cargar_padron


# ── métrica ──────────────────────────────────────────────────────────────────

def calcular_log_monto(df):
    """
    log_monto = log1p(monto_total acumulado por CUIT), SOLO sobre CUITs con
    monto positivo.

    La red de monto modela ESCALA económica: un CUIT sin monto positivo no
    tiene escala medible ni entidad compartible, así que se EXCLUYE del grafo
    en vez de agregarlo como nodo aislado. Esos CUITs siguen apareciendo en
    features.csv con features de red = NaN, que el modelo imputa downstream.
    """
    df_pos = df[df["monto"] > 0].copy()

    agg = df_pos.groupby("cuit").agg(
        monto_total=("monto", "sum"),
        monto_max=("monto", "max"),
        n_periodos=("periodo", "nunique"),
    )
    agg["log_monto"] = np.log1p(agg["monto_total"])

    n_total = df["cuit"].nunique()
    n_excluidos = n_total - len(agg)
    print(
        f"\nNodos (CUITs con monto>0): {len(agg):,} / {n_total:,} "
        f"| excluidos sin monto positivo: {n_excluidos:,}"
    )
    print(
        f"log_monto — min: {agg['log_monto'].min():.3f} "
        f"| max: {agg['log_monto'].max():.3f} "
        f"| media: {agg['log_monto'].mean():.3f}"
    )
    print(agg["log_monto"].describe().round(3).to_string())
    return agg


# ── grafo ────────────────────────────────────────────────────────────────────

def pares_banco_comun(cuits, ent_map):
    """Índices (ii, jj) de pares que comparten >= 1 entidad, y nº compartidas."""
    size = defaultdict(int)
    for c in cuits:
        for e in ent_map.get(c, ()):
            size[e] += 1
    ents = [e for e, n in size.items() if n >= 2]
    eidx = {e: i for i, e in enumerate(ents)}

    rows, cols = [], []
    for i, c in enumerate(cuits):
        for e in ent_map.get(c, ()):
            j = eidx.get(e)
            if j is not None:
                rows.append(i)
                cols.append(j)

    B = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(cuits), len(ents))
    )
    inter = sparse.triu(B @ B.T, k=1).tocoo()
    return inter.row, inter.col, inter.data.astype(int)


def construir_grafo(metricas, ent_map, delta_max=config.DELTA_MAX):
    cuits = list(metricas.index)
    mp = metricas["log_monto"].to_numpy()

    G = nx.Graph()
    for c in cuits:
        G.add_node(c, **{k: round(float(v), 4) for k, v in metricas.loc[c].items()})

    ii, jj, n_ents = pares_banco_comun(cuits, ent_map)
    if ii.size == 0:
        print("Sin pares con banco común.")
        return G

    diff = np.abs(mp[ii] - mp[jj])
    # Estricto (<): en el borde diff==delta_max el peso sería 0, y una arista de
    # peso 0 ensucia PageRank/comunidades/medias ponderadas sin aportar nada.
    mask = diff < delta_max
    print(
        f"Pares banco común: {mask.size:,} | con |Δ log_monto| < {delta_max}: {mask.sum():,}"
    )

    # peso: similitud normalizada en (0, 1]
    pesos = 1.0 - diff[mask] / delta_max

    G.add_edges_from(
        (
            cuits[i],
            cuits[j],
            {
                "weight": round(float(p), 4),
                "delta_mp": round(float(d), 4),
                "n_entidades": int(ne),
            },
        )
        for i, j, p, d, ne in zip(ii[mask], jj[mask], pesos, diff[mask], n_ents[mask])
    )
    print(f"Aristas: {G.number_of_edges():,}")
    n_aislados = sum(1 for _, d in G.degree() if d == 0)
    print(
        f"Nodos aislados (grado 0, su entidad no se comparte dentro del umbral): "
        f"{n_aislados:,} / {G.number_of_nodes():,}"
    )
    return G


def enriquecer(G, padron):
    """Adjunta atributos del padrón AFIP a cada nodo."""
    for n in G.nodes():
        a = padron.get(str(n), {})
        for c in config.AFIP_COLS:
            G.nodes[n][c] = a.get(c, "")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    config.asegurar_dirs(config.GRAPHS_DIR)
    df = cargar_datos(config.CSV_MERGED, require_monto=True)
    metricas = calcular_log_monto(df)
    ent_map = df[df["monto"] > 0].groupby("cuit")["entidad"].agg(set).to_dict()
    G = construir_grafo(metricas, ent_map)
    enriquecer(G, cargar_padron(config.PADRON_MONTO))
    nx.write_gexf(G, str(config.RED_MONTO_GEXF))
    print(f"\nGEXF: {config.RED_MONTO_GEXF}")


if __name__ == "__main__":
    main()
