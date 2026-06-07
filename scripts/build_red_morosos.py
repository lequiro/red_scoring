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
CSV = os.path.normpath(os.path.join(AQUI, "..", "data", "bcra_merged.csv"))
PADRON = os.path.normpath(os.path.join(AQUI, "..", "data", "padron_afip.csv"))

# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")
OUT = os.path.join(GRAPHS_DIR, "red_morosos.gexf")

FEATURES = [
    "peor_situacion",
    "situacion_final",
    "n_periodos_mora",
    "racha_max_mora",
    "tendencia",
    "n_transiciones",
    "volatilidad",
]

NEWMAN_MIN = 0.015  # umbral mínimo de peso normalizado; aristas por debajo se descartan (el que funciona es 0.015)
AFIP_COLS = [
    "denominacion",
    "estado_afip",
    "condicion_iva",
    "cat_ganancias",
    "empleador",
    "monotributo",
    "act_monotributo",
]


def _periodo_a_int(serie: pd.Series) -> pd.Series:
    """
    Convierte 'periodo' a entero AAAAMM de forma robusta al formato.

    Acepta 'YYYY-MM' y 'YYYYMM' (y cualquier separador): se quitan los
    no-dígitos antes de convertir. Evita el bug silencioso de
    pd.to_numeric('2024-01') -> NaN, que descartaba todas las filas.
    El entero AAAAMM preserva el orden cronológico para racha/tendencia.
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
        if maxs == 0:
            clase = "sin_historial"  # nunca tuvo deuda activa (thin-file)
        elif maxs == 1:
            clase = "sano"  # siempre al corriente
        elif (a >= 5).any() or final >= 4:
            # situacion 5 (irrecuperable) y 6 (irrec. por disp. técnica)
            clase = "irrecuperable"
        elif final <= 1:
            clase = "recuperado"  # tuvo mora pero normalizó
        else:
            clase = "cronico"  # sigue en mora
        filas[cuit] = {
            "comportamiento": clase,
            "peor_situacion": maxs,
            "situacion_final": final,
            "n_periodos_mora": int((a >= 2).sum()),
            "racha_max_mora": _racha_max(a >= 2),
            "tendencia": round(float(np.polyfit(np.arange(a.size), a, 1)[0]), 4)
            if a.size > 1
            else 0.0,
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
    B = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(cuits), len(ents))
    )
    return B, w


def construir_grafo(perfil, ent_map):
    cuits = list(perfil.index)
    clase = perfil["comportamiento"].to_numpy()
    G = nx.Graph()
    for c, cl in zip(cuits, clase):
        G.add_node(c, comportamiento=cl)

    B, w = incidencia(cuits, ent_map)
    inter = sparse.triu(B @ B.T, k=1).tocoo()  # entidades compartidas
    if inter.nnz == 0:
        print("Sin aristas.")
        return G
    newman = (B.multiply(w) @ B.T).tocsr()  # peso Newman raw

    ii, jj, cnt = inter.row, inter.col, inter.data.astype(int)
    same = clase[ii] == clase[jj]  # misma clase
    ii, jj, cnt = ii[same], jj[same], cnt[same]
    raw = np.asarray(newman[ii, jj]).ravel()
    pesos = raw / raw.max() if raw.size and raw.max() > 0 else raw  # normalizar [0,1]

    mask = pesos >= NEWMAN_MIN
    ii, jj, cnt, pesos = ii[mask], jj[mask], cnt[mask], pesos[mask]
    print(
        f"Aristas pre-threshold: {mask.size} | post-threshold (>={NEWMAN_MIN}): {mask.sum()}"
    )

    G.add_edges_from(
        (cuits[i], cuits[j], {"weight": round(float(p), 4), "n_entidades": int(c)})
        for i, j, p, c in zip(ii, jj, pesos, cnt)
    )
    print(f"Aristas: {G.number_of_edges()}")
    n_aislados = sum(1 for _, d in G.degree() if d == 0)
    print(
        f"Nodos aislados (grado 0): {n_aislados:,} / {G.number_of_nodes():,} — "
        f"incluyen thin-files sin entidad compartida; quedan sin features de red."
    )
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
    ent_map = df[df["monto"] > 0].groupby("cuit")["entidad"].agg(set).to_dict()
    G = construir_grafo(perfil, ent_map)
    enriquecer(G, perfil, cargar_padron())
    nx.write_gexf(G, OUT)
    print(f"GEXF: {OUT}")


if __name__ == "__main__":
    main()
