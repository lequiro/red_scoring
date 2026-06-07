"""
merge_bcra_datasets.py
----------------------
Unifica bcra_dataset.csv, bcra_dataset_01.csv y deudas_bcra_padron.ndjson
en un único DataFrame con schema canónico.

Schema final:
    cuit, denominacion, periodo, entidad, monto_miles_pesos, situacion,
    refinanciaciones, situacion_juridica, proceso_judicial, en_revision,
    dias_atraso, recategorizacion_oblig, tipo_registro, fecha_consulta
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── 1. Cargar los dos CSVs ────────────────────────────────────────────────────


def load_csvs() -> pd.DataFrame:
    csv_files = [
        DATA_DIR / "bcra_dataset.csv",
        DATA_DIR / "bcra_dataset_01.csv",
    ]
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, dtype={"cuit": str, "periodo": str})
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["fecha_consulta"] = pd.to_datetime(combined["fecha_consulta"])
    # Los CSVs no tienen recategorizacion_oblig
    combined["recategorizacion_oblig"] = float("nan")
    return combined


# ── 2. Cargar y normalizar el NDJSON ─────────────────────────────────────────

NDJSON_RENAME = {
    "monto": "monto_miles_pesos",
    "diasAtraso": "dias_atraso",
    "refinanciaciones": "refinanciaciones",  # mismo nombre, distinto tipo
    "situacionJuridica": "situacion_juridica",
    "procesoJud": "proceso_judicial",
    "enRevision": "en_revision",
    "recategorizacionOblig": "recategorizacion_oblig",
}


def load_ndjson() -> pd.DataFrame:
    records = []
    with open(DATA_DIR / "deudas_bcra_padron.ndjson", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.rename(columns=NDJSON_RENAME)

    # Convertir booleans → float para consistencia con los CSVs
    bool_cols = [
        "refinanciaciones",
        "situacion_juridica",
        "proceso_judicial",
        "en_revision",
        "recategorizacion_oblig",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Columnas que no trae el NDJSON
    df["tipo_registro"] = "historial"  # el NDJSON solo tiene registros con deuda
    df["fecha_consulta"] = pd.NaT

    # Asegurar tipos
    df["cuit"] = df["cuit"].astype(str)
    df["periodo"] = df["periodo"].astype(str)

    return df


# ── 3. Merge ──────────────────────────────────────────────────────────────────


def merge() -> pd.DataFrame:
    csvs = load_csvs()
    ndjson = load_ndjson()

    # Columnas canónicas (orden final)
    COLS = [
        "cuit",
        "denominacion",
        "periodo",
        "entidad",
        "monto_miles_pesos",
        "situacion",
        "refinanciaciones",
        "situacion_juridica",
        "proceso_judicial",
        "en_revision",
        "dias_atraso",
        "recategorizacion_oblig",
        "tipo_registro",
        "fecha_consulta",
    ]

    all_df = pd.concat([csvs, ndjson], ignore_index=True)[COLS]

    # Deduplicar: clave natural = (cuit, periodo, entidad)
    # Para sin_datos: entidad está vacía → la clave es (cuit, tipo_registro='sin_datos')
    # Ante duplicados, preferir la fila con fecha_consulta más reciente.
    # Los registros del NDJSON (fecha_consulta=NaT) quedan como fallback.
    all_df = all_df.sort_values("fecha_consulta", ascending=False, na_position="last")
    dedup = all_df.drop_duplicates(subset=["cuit", "periodo", "entidad"], keep="first")
    dedup = dedup.reset_index(drop=True)

    return dedup


# ── 4. Stats rápidas ──────────────────────────────────────────────────────────


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 50}")
    print(f"Total filas:          {len(df):>10,}")
    print(f"CUITs únicos:         {df['cuit'].nunique():>10,}")
    print(f"\nDistribución tipo_registro:")
    print(df["tipo_registro"].value_counts().to_string())
    print(f"\nDistribución situacion (historial):")
    hist = df[df["tipo_registro"] == "historial"]["situacion"]
    print(hist.value_counts().sort_index().to_string())
    print(f"\nCUITs con recategorizacion_oblig=1.0 (solo NDJSON):")
    print(f"  {(df['recategorizacion_oblig'] == 1.0).sum()}")
    print(f"{'=' * 50}\n")


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando CSVs...")
    csvs = load_csvs()
    print(f"  bcra_dataset*.csv: {len(csvs):,} filas")

    print("Cargando NDJSON...")
    ndjson = load_ndjson()
    print(f"  deudas_bcra_padron.ndjson: {len(ndjson):,} filas")

    print("Mergeando y deduplicando...")
    merged = merge()

    print_stats(merged)

    out_path = DATA_DIR / "bcra_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"Guardado en: {out_path}")
