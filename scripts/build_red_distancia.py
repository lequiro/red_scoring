#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Red ALTERNATIVA de CUITs: aristas por SIMILITUD CONTINUA de comportamiento
(distancia en el espacio de features), para comparar contra la red por
igualdad de clase (build_red_morosos.py).

Definicion de la red:
  - Nodos: TODOS los CUITs con datos reales (igual que la otra red).
  - Candidatos a arista: pares que comparten >= 1 entidad.
  - De esos, hay arista si la SIMILITUD de comportamiento >= umbral tau.
  - Peso de arista = similitud (continua, en [0,1]).

Como se mide la similitud (lo conversado):
  1. Features numericas por CUIT (peor_situacion, situacion_final,
     n_periodos_mora, racha_max_mora, tendencia, n_transiciones, volatilidad).
  2. Normalizacion min-max por feature (si no, la racha domina y la tendencia
     desaparece).
  3. Distancia euclidea entre vectores normalizados (NO cosine: aca la
     magnitud importa).
  4. Similitud = kernel gaussiano:  sim = exp(-d^2 / (2*sigma^2)),  con
     sigma = mediana de las distancias entre pares candidatos (heuristica
     estandar). sim en [0,1], 1 = identicos.
  5. tau = percentil P de las similitudes de los pares candidatos (data-driven,
     default P=85). El peso queda exportado, asi tambien podes mover el corte
     en Gephi con el slider.

Salida: output/red_distancia.gexf

Uso:
    python3 build_red_distancia.py
    python3 build_red_distancia.py --percentil 90
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx

# Features usadas para medir similitud de comportamiento.
FEATURES = ["peor_situacion", "situacion_final", "n_periodos_mora",
            "racha_max_mora", "tendencia", "n_transiciones", "volatilidad"]


def parse_args():
    aqui = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Red de CUITs por similitud continua.")
    p.add_argument("--csv", default=os.path.normpath(os.path.join(aqui, "..", "data", "bcra_dataset.csv")))
    p.add_argument("--padron", default=os.path.normpath(os.path.join(aqui, "..", "data", "padron_afip_1.csv")))
    p.add_argument("--outdir", default=os.path.normpath(os.path.join(aqui, "..", "output")))
    p.add_argument("--percentil", type=float, default=85.0,
                   help="Percentil de similitud para fijar tau (default 85).")
    p.add_argument("--solo-malos", action="store_true", default=False,
                   help="Incluir solo nodos cronico + irrecuperable (default: todos).")
    return p.parse_args()


# --- Carga y perfil (identicos a build_red_morosos.py, para comparar igual) -- #
def cargar_datos(csv_path):
    df = pd.read_csv(csv_path, dtype={"cuit": str})
    monto_col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={monto_col: "monto"})
    for col in ["periodo", "situacion", "monto", "dias_atraso"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["entidad"].notna() & (df["entidad"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["periodo", "situacion", "monto"])
    df["cuit"] = df["cuit"].astype(str).str.strip()
    print(f"Filas con datos reales: {len(df):,} | CUITs: {df['cuit'].nunique():,}")
    return df


def _racha_max(mask):
    mejor = act = 0
    for v in mask:
        act = act + 1 if v else 0
        mejor = max(mejor, act)
    return mejor


def _clasificar(s):
    maxs, final = int(s.max()), int(s[-1])
    if maxs <= 1:
        return "sano"
    if (s == 5).any() or final >= 4:
        return "irrecuperable"
    if final == 0:
        return "salida"
    if final == 1:
        return "recurrente"
    return "cronico"


def perfil_comportamiento(df):
    filas = {}
    for cuit, g in df.groupby("cuit"):
        serie = g.groupby("periodo")["situacion"].max().sort_index()
        s = serie.to_numpy().astype(int)
        if s.size == 0:
            continue
        tendencia = float(np.polyfit(np.arange(s.size), s, 1)[0]) if s.size > 1 else 0.0
        filas[cuit] = {
            "comportamiento": _clasificar(s),
            "peor_situacion": int(s.max()),
            "situacion_final": int(s[-1]),
            "n_periodos_mora": int((s >= 2).sum()),
            "racha_max_mora": int(_racha_max((s >= 2) & (s <= 4))),
            "tendencia": round(tendencia, 4),
            "n_transiciones": int((np.diff(s) != 0).sum()),
            "volatilidad": round(float(s.std()), 4),
        }
    perfil = pd.DataFrame.from_dict(filas, orient="index")
    print("\nDistribucion de comportamiento:")
    print(perfil["comportamiento"].value_counts().to_string())
    return perfil


def calcular_scores(df):
    num = (df["situacion"] * df["monto"]).groupby(df["cuit"]).sum()
    den = df.groupby("cuit")["monto"].sum() * 5.0
    score = (num / den).where(den > 0, 0.0).fillna(0.0).clip(0.0, 1.0)
    score.name = "score"
    return score


def entidades_por_cuit(df):
    sub = df[df["monto"] > 0]
    return sub.groupby("cuit")["entidad"].apply(lambda s: set(s.dropna())).to_dict()


# --------------------------------------------------------------------------- #
# Grafo por similitud continua                                                 #
# --------------------------------------------------------------------------- #
def matriz_normalizada(perfil):
    """Min-max por feature. Devuelve (array N x F, indice cuit->fila)."""
    X = perfil[FEATURES].to_numpy(dtype=float)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    rango = np.where((maxs - mins) == 0, 1.0, maxs - mins)  # evita /0
    Xn = (X - mins) / rango
    idx = {cuit: i for i, cuit in enumerate(perfil.index)}
    return Xn, idx


def construir_grafo_distancia(perfil, ent_map, percentil, solo_malos=False):
    cuits = list(perfil.index)
    if solo_malos:
        cuits = [c for c in cuits if perfil.loc[c, "comportamiento"] in ("cronico", "irrecuperable")]
    cuits_set = set(cuits)
    print(f"\nNodos en la red: {len(cuits_set)}")

    G = nx.Graph()
    for c in cuits_set:
        G.add_node(c, comportamiento=perfil.loc[c, "comportamiento"])

    Xn, idx = matriz_normalizada(perfil)

    # Pares candidatos = comparten >= 1 entidad. Contamos entidades compartidas.
    ent_index = {}
    for c in cuits_set:
        for ent in ent_map.get(c, set()):
            ent_index.setdefault(ent, []).append(c)

    pares = {}
    for ent, lista in ent_index.items():
        if len(lista) < 2:
            continue
        for a, b in combinations(sorted(lista), 2):
            pares[(a, b)] = pares.get((a, b), 0) + 1

    if not pares:
        print("No hay pares que compartan entidad.")
        return G

    ab = list(pares.keys())
    ia = np.array([idx[a] for a, _ in ab])
    ib = np.array([idx[b] for _, b in ab])
    dist = np.linalg.norm(Xn[ia] - Xn[ib], axis=1)

    sigma = np.median(dist)
    if sigma == 0:
        sigma = 1.0
    sim = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    tau = np.percentile(sim, percentil)

    print(f"Pares candidatos (comparten entidad): {len(ab):,}")
    print(f"sigma (mediana de distancias): {sigma:.4f}")
    print(f"tau (percentil {percentil:g} de similitud): {tau:.4f}")
    print(f"Similitud  min/median/max: {sim.min():.3f} / {np.median(sim):.3f} / {sim.max():.3f}")

    n = 0
    for (a, b), s, w in zip(ab, sim, [pares[p] for p in ab]):
        if s >= tau:
            G.add_edge(a, b, weight=round(float(s), 4), n_entidades=int(w))
            n += 1
    print(f"Aristas (sim >= tau): {n}")
    return G


def cargar_padron(padron_path):
    if not padron_path or not os.path.exists(padron_path):
        print(f"\nPadron AFIP no encontrado ({padron_path}); se omiten atributos.")
        return {}
    cols = ["denominacion", "estado_afip", "condicion_iva", "cat_ganancias",
            "empleador", "monotributo", "act_monotributo"]
    df = pd.read_csv(padron_path, dtype=str).fillna("")
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df = df.drop_duplicates(subset="cuit", keep="first").set_index("cuit")
    return {c: {k: str(r[k]).strip() for k in cols if k in df.columns}
            for c, r in df.iterrows()}


def enriquecer_nodos(G, perfil, score, padron):
    afip_cols = ["denominacion", "estado_afip", "condicion_iva", "cat_ganancias",
                 "empleador", "monotributo", "act_monotributo"]
    match = 0
    for n in G.nodes():
        for f in FEATURES:
            G.nodes[n][f] = perfil.loc[n, f]
        G.nodes[n]["score"] = float(score.get(n, 0.0))
        attrs = padron.get(str(n))
        if attrs:
            match += 1
        for c in afip_cols:
            G.nodes[n][c] = (attrs or {}).get(c, "")
    if padron:
        print(f"Nodos con match en padron AFIP: {match}/{G.number_of_nodes()}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = cargar_datos(args.csv)
    perfil = perfil_comportamiento(df)
    score = calcular_scores(df)
    ent_map = entidades_por_cuit(df)

    G = construir_grafo_distancia(perfil, ent_map, args.percentil, solo_malos=args.solo_malos)
    enriquecer_nodos(G, perfil, score, cargar_padron(args.padron))

    gexf = os.path.join(args.outdir, "red_distancia.gexf")
    nx.write_gexf(G, gexf)
    print(f"\nGEXF guardado: {gexf}")


if __name__ == "__main__":
    main()
