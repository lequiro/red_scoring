#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Red de CUITs (BCRA).
  Nodos : todos los CUITs con datos reales.
  Arista: comparten >= 1 entidad Y misma clase de comportamiento.
  Peso  : Newman / TF-IDF normalizado [0,1] (los bancos raros pesan, los hubs no).
  Salida: output/red_morosos.gexf

Correr:  python3 build_red_morosos.py
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(AQUI, "..", "data", "bcra_dataset.csv"))
PADRON = os.path.normpath(os.path.join(AQUI, "..", "data", "padron_afip_1.csv"))
OUT = os.path.normpath(os.path.join(AQUI, "..", "output", "red_morosos.gexf"))

FEATURES = ["peor_situacion", "situacion_final", "n_periodos_mora",
            "racha_max_mora", "tendencia", "n_transiciones", "volatilidad"]
AFIP_COLS = ["denominacion", "estado_afip", "condicion_iva", "cat_ganancias",
             "empleador", "monotributo", "act_monotributo"]


def cargar_datos():
    df = pd.read_csv(CSV, dtype={"cuit": str})
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})
    for c in ["periodo", "situacion", "monto"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["entidad"].notna() & (df["entidad"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["periodo", "situacion", "monto"])
    df["cuit"] = df["cuit"].astype(str).str.strip()
    print(f"Filas reales: {len(df):,} | CUITs: {df['cuit'].nunique():,}")
    return df


def _racha_max(mask):
    """Racha mas larga de True consecutivos (vectorizado)."""
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def perfil_comportamiento(df):
    """Etiqueta + features por CUIT, sobre la peor situacion de cada periodo."""
    peor = df.groupby(["cuit", "periodo"])["situacion"].max().sort_index()
    filas = {}
    for cuit, s in peor.groupby(level=0):
        a = s.to_numpy().astype(int)
        maxs, final = int(a.max()), int(a[-1])
        if maxs <= 1:
            clase = "sano"
        elif (a == 5).any() or final >= 4:
            clase = "irrecuperable"
        elif final == 0:
            clase = "salida"
        elif final == 1:
            clase = "recurrente"
        else:
            clase = "cronico"
        filas[cuit] = {
            "comportamiento": clase,
            "peor_situacion": maxs,
            "situacion_final": final,
            "n_periodos_mora": int((a >= 2).sum()),
            "racha_max_mora": _racha_max((a >= 2) & (a <= 4)),
            "tendencia": round(float(np.polyfit(np.arange(a.size), a, 1)[0]), 4) if a.size > 1 else 0.0,
            "n_transiciones": int((np.diff(a) != 0).sum()),
            "volatilidad": round(float(a.std()), 4),
        }
    perfil = pd.DataFrame.from_dict(filas, orient="index")
    print("Clases:", perfil["comportamiento"].value_counts().to_dict())
    return perfil


def incidencia(cuits, ent_map):
    """Matriz sparse CUIT x entidad (solo entidades con >=2 CUITs) y pesos Newman."""
    size = defaultdict(int)
    for c in cuits:
        for e in ent_map.get(c, ()):
            size[e] += 1
    ents = [e for e, n in size.items() if n >= 2]
    eidx = {e: i for i, e in enumerate(ents)}
    w = np.array([1.0 / (size[e] - 1) for e in ents])  # aporte Newman por entidad

    rows, cols = [], []
    for i, c in enumerate(cuits):
        for e in ent_map.get(c, ()):
            j = eidx.get(e)
            if j is not None:
                rows.append(i)
                cols.append(j)
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(cuits), len(ents)))
    return B, w


def construir_grafo(perfil, ent_map):
    cuits = list(perfil.index)
    clase = perfil["comportamiento"].to_numpy()
    G = nx.Graph()
    for c, cl in zip(cuits, clase):
        G.add_node(c, comportamiento=cl)

    B, w = incidencia(cuits, ent_map)
    inter = sparse.triu(B @ B.T, k=1).tocoo()              # entidades compartidas
    if inter.nnz == 0:
        print("Sin aristas."); return G
    newman = (B.multiply(w) @ B.T).tocsr()                  # peso Newman raw

    ii, jj, cnt = inter.row, inter.col, inter.data.astype(int)
    same = clase[ii] == clase[jj]                           # misma clase
    ii, jj, cnt = ii[same], jj[same], cnt[same]
    raw = np.asarray(newman[ii, jj]).ravel()
    pesos = raw / raw.max() if raw.size and raw.max() > 0 else raw  # normalizar [0,1]

    G.add_edges_from(
        (cuits[i], cuits[j], {"weight": round(float(p), 4), "n_entidades": int(c)})
        for i, j, p, c in zip(ii, jj, pesos, cnt)
    )
    print(f"Aristas: {G.number_of_edges()}")
    return G


def enriquecer(G, perfil, padron):
    feats = perfil[FEATURES].to_dict("index")
    for n in G.nodes():
        G.nodes[n].update(feats[n])
        a = padron.get(str(n), {})
        for c in AFIP_COLS:
            G.nodes[n][c] = a.get(c, "")


def cargar_padron():
    if not os.path.exists(PADRON):
        return {}
    df = pd.read_csv(PADRON, dtype=str).fillna("")
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df = df.drop_duplicates("cuit").set_index("cuit")
    keep = [c for c in AFIP_COLS if c in df.columns]
    return {c: r.to_dict() for c, r in df[keep].iterrows()}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = cargar_datos()
    perfil = perfil_comportamiento(df)
    ent_map = df[df["monto"] > 0].groupby("cuit")["entidad"].apply(set).to_dict()
    G = construir_grafo(perfil, ent_map)
    enriquecer(G, perfil, cargar_padron())
    nx.write_gexf(G, OUT)
    print(f"GEXF: {OUT}")


if __name__ == "__main__":
    main()
