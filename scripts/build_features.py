#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingeniería de features para el modelo de scoring.

  Fuentes:
    output/graphs/red_monto_comunidades.gexf → topología de red + log_monto, n_periodos, comunidad
    data/bcra_merged.csv                     → historial individual + features de entidad

  red_morosos quedó fuera del pipeline: las features [E] ya no se calculan.

  Salida:
    output/tables/features.csv

  Grupos de features:
    [A] Historial (CSV)     : peor_situacion, situacion_final, n_periodos_mora,
                              racha_max_mora, tendencia, n_transiciones, volatilidad
    [B] Escala (red_monto)  : log_monto, n_periodos, comunidad_escala (ordinal 0-3)
    [C] Vecindad monto      : degree_monto, pct_vecinos_sanos,
                              media_pond_sit_vecinos
    [D] Entidad (CSV)       : pct_monto_banco, tiene_fogAr, n_entidades_distintas

  Target:
    es_sano = 1 si comportamiento == "sano"
            = 0 para cronico | irrecuperable | recuperado | sin_historial

  ⚠ FUGA DE TARGET: el grupo [A] (peor_situacion, situacion_final,
    n_periodos_mora, ...) se deriva de la MISMA serie de `situacion` que define
    `comportamiento`/`es_sano`. De hecho sano ⟺ peor_situacion==1. Por eso [A]
    NO debe entrar como predictor: hace trivial cualquier modelo (AUC≈1) e
    invalida la comparación "red vs historial". Acá se calcula [A] solo como
    descriptor/EDA; modelo_scoring.py lo excluye de los sets de features.

Correr: python3 build_features.py
"""

import os

import numpy as np
import networkx as nx
import pandas as pd

AQUI          = os.path.dirname(os.path.abspath(__file__))
CSV           = os.path.normpath(os.path.join(AQUI, "..", "data", "bcra_merged.csv"))

# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR       = os.path.normpath(os.path.join(AQUI, "..", "output"))
GRAPHS_DIR    = os.path.join(OUT_DIR, "graphs")
TABLES_DIR    = os.path.join(OUT_DIR, "tables")
GEXF_MONTO    = os.path.join(GRAPHS_DIR, "red_monto_comunidades.gexf")
OUT_CSV       = os.path.join(TABLES_DIR, "features.csv")


# ── 1. carga CSV ───────────────────────────────────────────────────────────────

def _periodo_a_int(serie: pd.Series) -> pd.Series:
    """
    Convierte 'periodo' a entero AAAAMM robusto al formato ('YYYY-MM' o 'YYYYMM').
    Evita que pd.to_numeric('2024-01') -> NaN descarte todas las filas.
    """
    limpio = serie.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_numeric(limpio, errors="coerce")


def cargar_datos():
    df = pd.read_csv(CSV, dtype={"cuit": str})
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})
    n0 = len(df)
    df["periodo"] = _periodo_a_int(df["periodo"])
    df["situacion"] = pd.to_numeric(df["situacion"], errors="coerce")
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df = df[df["entidad"].notna() & (df["entidad"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["periodo", "situacion"])
    df["periodo"] = df["periodo"].astype(int)
    df["monto"] = df["monto"].fillna(0)
    df["cuit"] = df["cuit"].astype(str).str.strip()
    print(f"Filas: {n0:,} -> {len(df):,} tras limpieza | CUITs: {df['cuit'].nunique():,}")
    return df


# ── 2. historial por CUIT (features [A] + target) ────────────────────────────

def _racha_max(mask):
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def perfil_comportamiento(df):
    """
    Clasifica cada CUIT por su trayectoria de `situacion` (etiqueta de
    comportamiento) y calcula las features de historial [A]. Es la única fuente
    de `comportamiento` en el pipeline (red_morosos quedó fuera).
    """
    peor = df.groupby(["cuit", "periodo"])["situacion"].max().sort_index()
    filas = {}
    for cuit, s in peor.groupby(level=0):
        a = s.to_numpy().astype(int)
        maxs, final = int(a.max()), int(a[-1])
        if maxs == 0:
            clase = "sin_historial"
        elif maxs == 1:
            clase = "sano"
        elif (a >= 5).any() or final >= 4:
            # situacion 5 (irrecuperable) y 6 (irrec. por disp. técnica)
            clase = "irrecuperable"
        elif final <= 1:
            clase = "recuperado"
        else:
            clase = "cronico"
        filas[cuit] = {
            "comportamiento": clase,
            "peor_situacion":  maxs,
            "situacion_final": final,
            "n_periodos_mora": int((a >= 2).sum()),
            "racha_max_mora":  _racha_max(a >= 2),
            "tendencia":       round(float(np.polyfit(np.arange(a.size), a, 1)[0]), 4)
                               if a.size > 1 else 0.0,
            "n_transiciones":  int((np.diff(a) != 0).sum()),
            "volatilidad":     round(float(a.std()), 4),
        }
    perfil = pd.DataFrame.from_dict(filas, orient="index")
    print("Clases:", perfil["comportamiento"].value_counts().to_dict())
    return perfil


# ── 3. features de vecindad sobre red_monto (features [C]) ───────────────────

def features_vecindad(G, perfil):
    """
    Para cada nodo en red_monto:
      - degree_monto            : grado en la red de escala
      - pct_vecinos_sanos       : fracción de vecinos con comportamiento 'sano'
      - media_pond_sit_vecinos  : media ponderada de peor_situacion de vecinos
                                  (peso = arista de similitud de monto)

    Los atributos de comportamiento vienen del perfil calculado sobre el CSV,
    no del GEXF — así la red y el target son independientes.
    """
    records = {}
    for n in G.nodes():
        vecinos = list(G.neighbors(n))
        deg = len(vecinos)

        if deg == 0:
            records[n] = {
                "degree_monto": 0,
                "pct_vecinos_sanos": np.nan,
                "media_pond_sit_vecinos": np.nan,
            }
            continue

        pesos = np.array([G[n][v].get("weight", 1.0) for v in vecinos])
        comportamientos = [perfil.loc[v, "comportamiento"]
                           if v in perfil.index else "" for v in vecinos]
        n_sanos = sum(1 for c in comportamientos if c == "sano")

        sits = np.array([
            float(perfil.loc[v, "peor_situacion"])
            if v in perfil.index else np.nan
            for v in vecinos
        ])
        valid = ~np.isnan(sits)
        if valid.sum() == 0:
            media_pond = np.nan
        elif pesos[valid].sum() > 0:
            media_pond = float(np.average(sits[valid], weights=pesos[valid]))
        else:
            # red_monto puede emitir weight=0 (cuando |Δ|==DELTA_MAX); si todos
            # los pesos válidos son 0, np.average daría ZeroDivisionError -> media simple
            media_pond = float(np.mean(sits[valid]))

        records[n] = {
            "degree_monto": deg,
            "pct_vecinos_sanos": n_sanos / deg,
            "media_pond_sit_vecinos": media_pond,
        }

    return pd.DataFrame.from_dict(records, orient="index")


# ── 4. features de entidad (features [D]) ─────────────────────────────────────

def features_entidad(df):
    df_pos = df[df["monto"] > 0].copy()
    df_pos["es_banco"] = df_pos["entidad"].str.contains(
        r"\bBANCO\b", case=False, na=False, regex=True
    )
    df_pos["es_fogar"] = df_pos["entidad"].str.contains(
        "GARANT", case=False, na=False
    )

    monto_total = df_pos.groupby("cuit")["monto"].sum()
    monto_banco = df_pos[df_pos["es_banco"]].groupby("cuit")["monto"].sum()
    tiene_fogar = df_pos[df_pos["es_fogar"]].groupby("cuit").size().gt(0)
    n_ents      = df_pos.groupby("cuit")["entidad"].nunique().rename("n_entidades_distintas")

    feats = pd.concat([monto_total.rename("mt"), monto_banco.rename("mb"),
                       tiene_fogar.rename("tiene_fogAr"), n_ents], axis=1)
    feats["pct_monto_banco"] = (feats["mb"] / feats["mt"]).fillna(0.0)
    feats["tiene_fogAr"]     = feats["tiene_fogAr"].fillna(False).astype(int)
    return feats[["pct_monto_banco", "tiene_fogAr", "n_entidades_distintas"]]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print("Cargando CSV ...")
    df = cargar_datos()

    print("\nCalculando historial por CUIT ...")
    perfil = perfil_comportamiento(df)

    print("\nCargando red_monto_comunidades.gexf ...")
    G = nx.read_gexf(GEXF_MONTO)
    print(f"Nodos: {G.number_of_nodes():,} | Aristas: {G.number_of_edges():,}")

    # features [B] — escala desde GEXF
    df_escala = pd.DataFrame(
        [{"cuit": n,
          "log_monto":  float(d.get("log_monto", np.nan)),
          "n_periodos": float(d.get("n_periodos", np.nan)),
          "comunidad":  d.get("comunidad", np.nan)}
         for n, d in G.nodes(data=True)]
    ).set_index("cuit")
    df_escala["comunidad"] = pd.to_numeric(df_escala["comunidad"], errors="coerce")

    # Rango ordinal de escala: rank de cada comunidad por su log_monto medio
    # Cubre las 39 comunidades, no solo las top-4
    media_por_comunidad = df_escala.groupby("comunidad")["log_monto"].mean()
    rango_comunidad = media_por_comunidad.rank(method="first").astype(int) - 1
    df_escala["comunidad_escala"] = df_escala["comunidad"].map(rango_comunidad)
    print(f"Comunidades con escala asignada: {df_escala['comunidad_escala'].notna().sum()} "
          f"/ {len(df_escala)} nodos | NaNs: {df_escala['comunidad_escala'].isna().sum()}")

    print("\nCalculando features de vecindad ...")
    df_red = features_vecindad(G, perfil)

    print("Calculando features de entidad ...")
    df_ent = features_entidad(df)

    # ── join ──────────────────────────────────────────────────────────────────
    FEATS_A = ["peor_situacion", "situacion_final", "n_periodos_mora",
               "racha_max_mora", "tendencia", "n_transiciones", "volatilidad",
               "comportamiento"]
    FEATS_B = ["log_monto", "n_periodos", "comunidad_escala"]

    df_final = (
        perfil[FEATS_A]
        .join(df_escala[FEATS_B], how="left")
        .join(df_red,             how="left")
        .join(df_ent,             how="left")
    )

    df_final["es_sano"] = (df_final["comportamiento"] == "sano").astype(int)

    print(f"\nShape: {df_final.shape}")
    print(f"Target — sano: {df_final['es_sano'].sum()} / {len(df_final)} "
          f"({100 * df_final['es_sano'].mean():.1f}%)")
    print(f"\nNaNs por columna:\n{df_final.isnull().sum().to_string()}")

    df_final.to_csv(OUT_CSV)
    print(f"\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
