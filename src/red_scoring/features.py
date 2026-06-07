"""
Ingeniería de features para el modelo de scoring.

  Fuentes:
    output/graphs/red_monto_comunidades.gexf → topología + log_monto, n_periodos, comunidad
    data/bcra_merged.csv                     → historial individual + features de entidad

  Grupos:
    [A] Historial (CSV)     : ver behavior.perfil_comportamiento  (descriptor, no predictor)
    [B] Escala (red_monto)  : log_monto, n_periodos, comunidad_escala
    [C] Vecindad monto      : degree_monto, pct_vecinos_sanos, media_pond_sit_vecinos
    [D] Entidad (CSV)       : pct_monto_banco, tiene_fogAr, n_entidades_distintas

  Target: es_sano = 1 si comportamiento == "sano", si no 0.

  Salida: output/tables/features.csv
"""

import numpy as np
import networkx as nx
import pandas as pd

from . import config
from .io import cargar_datos
from .behavior import perfil_comportamiento

FEATS_A = ["peor_situacion", "situacion_final", "n_periodos_mora",
           "racha_max_mora", "tendencia", "n_transiciones", "volatilidad",
           "comportamiento"]
FEATS_B = ["log_monto", "n_periodos", "comunidad_escala"]


# ── features [C] — vecindad sobre red_monto ──────────────────────────────────

def features_vecindad(G, perfil) -> pd.DataFrame:
    """
    Por nodo en red_monto:
      - degree_monto           : grado en la red de escala
      - pct_vecinos_sanos      : fracción de vecinos con comportamiento 'sano'
      - media_pond_sit_vecinos : media ponderada de peor_situacion de vecinos
                                 (peso = similitud de monto de la arista)

    El comportamiento de los vecinos viene del perfil (CSV), no del GEXF, así la
    red y el target son independientes.
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
            # red_monto puede emitir weight=0; si todos los pesos válidos son 0,
            # np.average daría ZeroDivisionError -> media simple
            media_pond = float(np.mean(sits[valid]))

        records[n] = {
            "degree_monto": deg,
            "pct_vecinos_sanos": n_sanos / deg,
            "media_pond_sit_vecinos": media_pond,
        }

    return pd.DataFrame.from_dict(records, orient="index")


# ── features [D] — entidad financiera ────────────────────────────────────────

def features_entidad(df) -> pd.DataFrame:
    df_pos = df[df["monto"] > 0].copy()
    df_pos["es_banco"] = df_pos["entidad"].astype(str).str.contains(
        r"\bBANCO\b", case=False, na=False, regex=True
    )
    df_pos["es_fogar"] = df_pos["entidad"].astype(str).str.contains(
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


# ── features [B] — escala desde el GEXF de comunidades ───────────────────────

def features_escala(G) -> pd.DataFrame:
    """Extrae log_monto, n_periodos y un rango ordinal de comunidad por escala."""
    df_escala = pd.DataFrame(
        [{"cuit": n,
          "log_monto":  float(d.get("log_monto", np.nan)),
          "n_periodos": float(d.get("n_periodos", np.nan)),
          "comunidad":  d.get("comunidad", np.nan)}
         for n, d in G.nodes(data=True)]
    ).set_index("cuit")
    df_escala["comunidad"] = pd.to_numeric(df_escala["comunidad"], errors="coerce")

    # rango ordinal de escala: cada comunidad por su log_monto medio
    media_por_comunidad = df_escala.groupby("comunidad")["log_monto"].mean()
    rango_comunidad = media_por_comunidad.rank(method="first").astype(int) - 1
    df_escala["comunidad_escala"] = df_escala["comunidad"].map(rango_comunidad)
    print(f"Comunidades con escala asignada: {df_escala['comunidad_escala'].notna().sum()} "
          f"/ {len(df_escala)} nodos | NaNs: {df_escala['comunidad_escala'].isna().sum()}")
    return df_escala


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    config.asegurar_dirs(config.TABLES_DIR)

    print("Cargando CSV ...")
    df = cargar_datos(config.CSV_MERGED, require_monto=False)

    print("\nCalculando historial por CUIT ...")
    perfil = perfil_comportamiento(df)

    print(f"\nCargando {config.RED_MONTO_COMUNIDADES_GEXF.name} ...")
    G = nx.read_gexf(str(config.RED_MONTO_COMUNIDADES_GEXF))
    print(f"Nodos: {G.number_of_nodes():,} | Aristas: {G.number_of_edges():,}")

    df_escala = features_escala(G)

    print("\nCalculando features de vecindad ...")
    df_red = features_vecindad(G, perfil)

    print("Calculando features de entidad ...")
    df_ent = features_entidad(df)

    df_final = (
        perfil[FEATS_A]
        .join(df_escala[FEATS_B], how="left")
        .join(df_red,             how="left")
        .join(df_ent,             how="left")
    )
    df_final["es_sano"] = (df_final["comportamiento"] == "sano").astype(int)
    df_final.index.name = "cuit"  # cuit explícito en el CSV (evita que se relea como int)

    print(f"\nShape: {df_final.shape}")
    print(f"Target — sano: {df_final['es_sano'].sum()} / {len(df_final)} "
          f"({100 * df_final['es_sano'].mean():.1f}%)")
    print(f"\nNaNs por columna:\n{df_final.isnull().sum().to_string()}")

    df_final.to_csv(str(config.FEATURES_CSV))
    print(f"\nCSV: {config.FEATURES_CSV}")


if __name__ == "__main__":
    main()
