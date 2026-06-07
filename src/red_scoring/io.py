"""
Carga y normalización de datos. Fuente única de verdad para parsear el CSV
mergeado del BCRA y el padrón AFIP — antes esta lógica estaba duplicada en
varios scripts.
"""

import networkx as nx
import pandas as pd

from . import config


def periodo_a_int(serie: pd.Series) -> pd.Series:
    """
    Convierte 'periodo' a entero AAAAMM de forma robusta al formato.

    Acepta 'YYYY-MM' y 'YYYYMM' (y cualquier separador): se quitan los
    no-dígitos antes de convertir. Evita el bug silencioso de hacer
    pd.to_numeric('2024-01') -> NaN, que descartaba todas las filas.
    El entero AAAAMM preserva el orden cronológico (útil para racha/tendencia).
    """
    limpio = serie.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_numeric(limpio, errors="coerce")


def cargar_datos(
    csv_path=config.CSV_MERGED,
    require_monto: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Carga bcra_merged.csv y lo normaliza para el pipeline.

    - 'monto_miles_pesos' (o 'monto') -> columna 'monto'
    - 'periodo' -> entero AAAAMM (ver periodo_a_int)
    - filtra filas sin entidad
    - require_monto=True : descarta filas con monto nulo (uso: grafo de escala)
      require_monto=False: conserva esas filas con monto=0 (uso: features/perfil)
    """
    df = pd.read_csv(csv_path, dtype={"cuit": str})
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})

    n0 = len(df)
    df["periodo"] = periodo_a_int(df["periodo"])
    df["situacion"] = pd.to_numeric(df["situacion"], errors="coerce")
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")

    df = df[df["entidad"].notna() & (df["entidad"].astype(str).str.strip() != "")]
    subset = ["periodo", "situacion", "monto"] if require_monto else ["periodo", "situacion"]
    df = df.dropna(subset=subset)

    df["periodo"] = df["periodo"].astype(int)
    if not require_monto:
        df["monto"] = df["monto"].fillna(0)
    df["cuit"] = df["cuit"].astype(str).str.strip()

    if verbose:
        print(f"Filas: {n0:,} -> {len(df):,} tras limpieza | CUITs: {df['cuit'].nunique():,}")
    return df


def cargar_csv_basico(csv_path=config.CSV_MERGED) -> pd.DataFrame:
    """
    Carga ligera para análisis por entidad: NO parsea periodo ni descarta por
    situacion. Solo normaliza monto/situacion a numérico y filtra entidad nula.
    """
    df = pd.read_csv(csv_path, dtype={"cuit": str})
    col = "monto_miles_pesos" if "monto_miles_pesos" in df.columns else "monto"
    df = df.rename(columns={col: "monto"})
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df["situacion"] = pd.to_numeric(df["situacion"], errors="coerce")
    return df[df["entidad"].notna()]


def gexf_nodos_a_df(path, attrs) -> pd.DataFrame:
    """Extrae atributos de nodos de un GEXF como DataFrame indexado por cuit."""
    G = nx.read_gexf(str(path))
    filas = [{"cuit": n, **{a: d.get(a) for a in attrs}} for n, d in G.nodes(data=True)]
    return pd.DataFrame(filas).set_index("cuit")


def cargar_padron(padron_path=config.PADRON_MONTO) -> dict:
    """
    Devuelve {cuit: {col_afip: valor}} desde el padrón AFIP.
    Si el archivo no existe, devuelve {} (el enriquecimiento queda vacío).
    """
    from pathlib import Path

    if not Path(padron_path).exists():
        return {}
    df = pd.read_csv(padron_path, dtype=str).fillna("")
    df["cuit"] = df["cuit"].astype(str).str.strip()
    df = df.drop_duplicates("cuit").set_index("cuit")
    keep = [c for c in config.AFIP_COLS if c in df.columns]
    return {c: r.to_dict() for c, r in df[keep].iterrows()}
