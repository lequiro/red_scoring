#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Red de CUITs — escala económica (log de monto total).

  Esta es la RED PRINCIPAL del proyecto (red_morosos quedó fuera del pipeline).

  Nodos      : CUITs con monto positivo (los que tienen escala medible).
  Condición 1: comparten >= 1 entidad financiera
  Condición 2: |log_monto_i - log_monto_j| < DELTA_MAX
  Métrica    : log1p(monto_total acumulado)
               captura escala económica con varianza distribuida en todo el rango
  Peso arista: 1 - |diff| / DELTA_MAX  →  (0, 1], más alto = más similares
  Salida     : output/graphs/red_monto.gexf

Correr: python3 build_red_monto.py
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(AQUI, "..", "data", "bcra_merged.csv"))
PADRON = os.path.normpath(os.path.join(AQUI, "..", "data", "padron_afip_1.csv"))

# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")
OUT = os.path.join(GRAPHS_DIR, "red_monto.gexf")

# Umbral de similitud de escala en espacio log-natural.
# OJO: 0.5 NO es "1 orden de magnitud". En log natural, |Δ|=0.5 equivale a un
# factor e^0.5 ≈ 1.65x entre los montos; un orden de magnitud sería ln(10) ≈ 2.30.
DELTA_MAX = 0.5  # máx. |Δ log_monto| para conectar (~1.65x de diferencia de monto)

AFIP_COLS = [
    "denominacion",
    "estado_afip",
    "condicion_iva",
    "cat_ganancias",
    "empleador",
    "monotributo",
    "act_monotributo",
]


# ── carga ──────────────────────────────────────────────────────────────────────


def _periodo_a_int(serie: pd.Series) -> pd.Series:
    """
    Convierte 'periodo' a entero AAAAMM de forma robusta al formato.

    Acepta 'YYYY-MM' y 'YYYYMM' (y cualquier separador): se quitan los
    no-dígitos antes de convertir. Esto evita el bug silencioso de hacer
    pd.to_numeric('2024-01') -> NaN, que descartaba todas las filas.
    """
    limpio = serie.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_numeric(limpio, errors="coerce")


def cargar_datos():
    df = pd.read_csv(CSV, dtype={"cuit": str})
    print("Columnas CSV:", df.columns.tolist())
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})
    n0 = len(df)
    df["periodo"] = _periodo_a_int(df["periodo"])
    for c in ["situacion", "monto"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["entidad"].notna() & (df["entidad"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["periodo", "situacion", "monto"])
    df["periodo"] = df["periodo"].astype(int)
    df["cuit"] = df["cuit"].astype(str).str.strip()
    print(f"Filas: {n0:,} -> {len(df):,} tras limpieza | CUITs: {df['cuit'].nunique():,}")
    return df


def cargar_padron():
    if not os.path.exists(PADRON):
        return {}
    df = pd.read_csv(PADRON, dtype=str).fillna("")
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df = df.drop_duplicates("cuit").set_index("cuit")
    keep = [c for c in AFIP_COLS if c in df.columns]
    return {c: r.to_dict() for c, r in df[keep].iterrows()}


# ── métrica ────────────────────────────────────────────────────────────────────


def calcular_log_monto(df):
    """
    log_monto = log1p(monto_total acumulado por CUIT), SOLO sobre CUITs con
    monto positivo.

    La red de monto modela ESCALA económica: un CUIT sin monto positivo no
    tiene escala medible ni entidad compartible, así que se EXCLUYE del grafo
    en vez de agregarlo como nodo aislado (comportamiento anterior, engañoso).
    Esos CUITs siguen apareciendo en features.csv con features de red = NaN,
    que el modelo imputa downstream.
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


# ── grafo ──────────────────────────────────────────────────────────────────────


def pares_banco_comun(cuits, ent_map):
    """Índices (ii, jj) de pares que comparten >= 1 entidad."""
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


def construir_grafo(metricas, ent_map):
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
    # Estricto (<): en el borde diff==DELTA_MAX el peso sería 0, y una arista de
    # peso 0 ensucia PageRank/comunidades/medias ponderadas sin aportar nada.
    mask = diff < DELTA_MAX
    print(
        f"Pares banco común: {mask.size:,} | con |Δ log_monto| < {DELTA_MAX}: {mask.sum():,}"
    )

    # peso: similitud normalizada en (0, 1]
    pesos = 1.0 - diff[mask] / DELTA_MAX

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
    for n in G.nodes():
        a = padron.get(str(n), {})
        for c in AFIP_COLS:
            G.nodes[n][c] = a.get(c, "")


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = cargar_datos()
    metricas = calcular_log_monto(df)
    ent_map = df[df["monto"] > 0].groupby("cuit")["entidad"].agg(set).to_dict()
    G = construir_grafo(metricas, ent_map)
    enriquecer(G, cargar_padron())
    nx.write_gexf(G, OUT)
    print(f"\nGEXF: {OUT}")


if __name__ == "__main__":
    main()
