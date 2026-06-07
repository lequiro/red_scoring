"""
bcra_fetch.py
Red de Scoring Crediticio — Módulo de ingesta de datos
======================================================
Fuentes:
  - Argentina Compra (ONC): semilla de CUITs de proveedores del estado
  - BCRA Central de Deudores: situación crediticia por CUIT
  - BCRA Estadísticas: tasas activas

Uso básico (Spyder, runcell):
    df_deudas, df_tasas = run_pipeline(n_cuits=50)
    describe_deudas(df_deudas)

Pipeline masivo desde padrón AFIP (entry point por defecto):
    run_pipeline_padron()
    # Lee CUITs de ../data/padron_afip.csv, retoma corridas parciales
    # automáticamente desde _checkpoint.txt, guarda en deudas_bcra_padron.ndjson
"""

import asyncio
import json
import logging
import random
import requests
import httpx
import pandas as pd
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── CONFIG ───────────────────────────────────────────────────────────────────

BCRA_BASE = "https://api.bcra.gob.ar"
AC_BASE = "https://api.argentinacompra.gob.ar"

HEADERS = {
    "User-Agent": "scoring-research/1.0",
    "Accept": "application/json",
}

DELAY_BCRA = 0.5  # segundos entre requests — no martillar la API
DELAY_AC = 0.3

# ── CONFIG PIPELINE MASIVO (padrón AFIP) ─────────────────────────────────────

MAX_CONCURRENT = 1  # requests simultáneos a BCRA — 1 = secuencial, sin riesgo de 429
REQUEST_DELAY = 1.2  # segundos entre requests (no bajar de 1.0)
MAX_RETRIES = 3  # reintentos ante HTTP 429 / 5xx antes de abandonar
SAVE_EVERY_N = 50  # flush a disco cada N CUITs procesados
LOG_EVERY_N = 100  # log de progreso cada N CUITs procesados
MAX_CUITS = 6000  # límite de CUITs a procesar por corrida (None = todos)

_SCRIPTS_DIR = Path(__file__).parent
PADRON_PATH = _SCRIPTS_DIR / "../data/padron_afip.csv"
OUTPUT_FILE = _SCRIPTS_DIR / "../data/deudas_bcra_padron.ndjson"
CHECKPOINT_FILE = _SCRIPTS_DIR / "../data/_checkpoint.txt"

# Fallback: CUITs conocidos para testing rápido sin Argentina Compra
# (son CUITs válidos en formato pero pueden no tener deudas registradas)
CUITS_TEST = [
    "30500010912",  # YPF SA
    "30546653569",  # Banco Nación
    "30708088932",  # Mercado Libre
    "30678774495",  # Telecom
    "30504349925",  # Arcor
    "30571400507",  # Grupo Clarín
    "30707429468",  # MercadoPago
    "30714024045",  # Naranja X
]

# IDs de variables BCRA relevantes para scoring
# Verificar en: GET /estadisticas/v2.0/principalesvariables
VARIABLES_TASAS = {
    6: "Tasa politica monetaria (%)",
    7: "BADLAR bancos privados (%)",
    8: "TM20 bancos privados (%)",
    29: "Tasa activa cartera general BNA (%)",
    30: "Tasa activa Adelantos cta cte BNA (%)",
}


# ── ARGENTINA COMPRA — fuente de CUITs ──────────────────────────────────────


def fetch_cuits_argentina_compra(n: int = 100) -> list[str]:
    """
    Obtiene hasta `n` CUITs únicos de proveedores del estado.

    Endpoint ONC v2 — si falla (cambio de API, downtime), cae al fallback.

    NOTA: Si el endpoint devuelve 404 o estructura inesperada, revisar:
      https://api.argentinacompra.gob.ar/  (swagger disponible ahí)
    """
    url = f"{AC_BASE}/api/v2/parametros/proveedores/"
    params = {"limit": min(n, 500), "offset": 0}

    try:
        r = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=15)
        r.raise_for_status()
        data = r.json()

        # La respuesta puede ser lista directa o envuelta en data/results
        items = (
            data
            if isinstance(data, list)
            else data.get("data", data.get("results", []))
        )

        cuits = []
        for item in items:
            # Posibles claves según versión de la API
            raw = str(
                item.get("cuit", item.get("nroCuit", item.get("cuil", "")))
            ).replace("-", "")
            if len(raw) == 11 and raw.isdigit():
                cuits.append(raw)

        cuits = list(dict.fromkeys(cuits))  # dedup preservando orden
        print(f"[ArgCompra] {len(cuits)} CUITs únicos obtenidos")
        return cuits[:n]

    except Exception as e:
        print(f"[ArgCompra] Error: {e}")
        print(f"[ArgCompra] Usando {len(CUITS_TEST)} CUITs de prueba (fallback)")
        return CUITS_TEST[:n]


# ── BCRA — Central de Deudores ───────────────────────────────────────────────


def fetch_deuda_bcra(cuit: str) -> Optional[dict]:
    """
    Consulta la situación crediticia de un CUIT.

    Respuesta esperada:
    {
      "status": 200,
      "results": {
        "identificacion": 20123456789,
        "denominacion": "NOMBRE PERSONA/EMPRESA",
        "periodos": [
          {
            "periodo": "2024-01",
            "entidades": [
              {
                "entidad": 11,
                "situacion": 1,        ← 1=normal, 2=riesgo bajo, ..., 6=irrecuperable
                "fechaSit1": "2020-01",
                "monto": 150,           ← en miles de pesos
                "diasAtrasoPago": 0,
                "refinanciaciones": false,
                "recategorizacionOblig": false,
                "situacionJuridica": false,
                "irrecuperables": false,
                "enRevision": false,
                "procesoJud": false
              }
            ]
          }
        ]
      }
    }

    Returns: dict con la respuesta cruda, o None si no hay datos.
    """
    url = f"{BCRA_BASE}/CentralDeDeudores/v1.0/Deudas/{cuit}"
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=10)
        if r.status_code == 404:
            return None  # sin deudas registradas — dato válido
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        print(f"  [BCRA] Timeout: {cuit}")
        return None
    except Exception as e:
        print(f"  [BCRA] Error {cuit}: {e}")
        return None


def parse_deuda(raw: dict, cuit: str) -> list[dict]:
    """
    Aplana la respuesta jerárquica del BCRA en filas planas.

    Jerarquía: cuit → periodos[] → entidades[]
    Una fila = un par (cuit, periodo, entidad financiera).

    El campo `situacion` es el ground truth del proyecto:
      1 = Normal
      2 = Riesgo bajo (hasta 90 días de atraso)
      3 = Riesgo medio (91–180 días)
      4 = Riesgo alto (181–365 días)
      5 = Irrecuperable
      6 = Irrecuperable por disposición técnica
    """
    rows = []
    results = raw.get("results", {})
    denominacion = results.get("denominacion", "")

    for periodo_obj in results.get("periodos", []):
        periodo = periodo_obj.get("periodo", "")
        for ent in periodo_obj.get("entidades", []):
            rows.append(
                {
                    "cuit": cuit,
                    "denominacion": denominacion,
                    "periodo": periodo,
                    "entidad": ent.get("entidad"),
                    "situacion": ent.get("situacion"),
                    "monto": ent.get("monto"),  # miles ARS
                    "diasAtraso": ent.get("diasAtrasoPago", 0),
                    "refinanciaciones": ent.get("refinanciaciones", False),
                    "recategorizacionOblig": ent.get("recategorizacionOblig", False),
                    "situacionJuridica": ent.get("situacionJuridica", False),
                    "enRevision": ent.get("enRevision", False),
                    "procesoJud": ent.get("procesoJud", False),
                }
            )
    return rows


# ── BCRA — Tasas activas ─────────────────────────────────────────────────────


def fetch_variables_bcra() -> pd.DataFrame:
    """
    Lista todas las principales variables del BCRA con sus IDs.
    Útil para identificar qué IDs usar en fetch_serie_variable().
    """
    url = f"{BCRA_BASE}/estadisticas/v2.0/principalesvariables"
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", data)
        df = pd.DataFrame(results if isinstance(results, list) else [results])
        print(f"[BCRA Variables] {len(df)} variables disponibles")
        return df
    except Exception as e:
        print(f"[BCRA Variables] Error: {e}")
        return pd.DataFrame()


def fetch_serie_variable(
    id_variable: int, desde: str = "2023-01-01", hasta: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Serie histórica de una variable BCRA por ID.

    Args:
        id_variable : ID numérico (ver VARIABLES_TASAS o fetch_variables_bcra())
        desde/hasta : formato "yyyy-MM-dd"

    Returns:
        DataFrame con columnas: fecha, valor, idVariable
    """
    url = f"{BCRA_BASE}/estadisticas/v2.0/DatosVariable/{id_variable}/{desde}/{hasta}"
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data.get("results", []))
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"])
            df["idVariable"] = id_variable
        print(f"[BCRA Serie {id_variable}] {len(df)} observaciones")
        return df
    except Exception as e:
        print(f"[BCRA Serie {id_variable}] Error: {e}")
        return pd.DataFrame()


def fetch_tasas_multiples(
    variables: dict = VARIABLES_TASAS,
    desde: str = "2023-01-01",
    hasta: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Descarga series de múltiples variables de tasas y las combina.
    Por defecto usa VARIABLES_TASAS (las más relevantes para scoring).
    """
    dfs = []
    for id_var, nombre in variables.items():
        df = fetch_serie_variable(id_var, desde, hasta)
        if not df.empty:
            df["nombre"] = nombre
            dfs.append(df)
        time.sleep(DELAY_BCRA)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ── PIPELINE MASIVO — async con checkpoint ────────────────────────────────────

logger = logging.getLogger(__name__)


def _load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Devuelve el set de CUITs ya procesados según el archivo de checkpoint."""
    if not checkpoint_file.exists():
        return set()
    with checkpoint_file.open("r") as f:
        return {line.strip() for line in f if line.strip()}


def _append_checkpoint(checkpoint_file: Path, cuits: list[str]) -> None:
    """Agrega CUITs al archivo de checkpoint (una línea por CUIT)."""
    with checkpoint_file.open("a") as f:
        for cuit in cuits:
            f.write(cuit + "\n")


def _append_rows_ndjson(output_file: Path, rows: list[dict]) -> None:
    """Escribe filas en formato NDJSON (append). Crea el archivo si no existe."""
    if not rows:
        return
    with output_file.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _fetch_one(
    cuit: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> tuple[str, Optional[dict], str]:
    """
    Consulta async la Central de Deudores para un CUIT.
    Reintenta ante HTTP 429 / 5xx con backoff exponencial + jitter.

    Returns:
        (cuit, raw_response_or_None, status)
        status ∈ {"ok", "sin_datos", "error_fatal"}
    """
    url = f"{BCRA_BASE}/CentralDeDeudores/v1.0/Deudas/{cuit}"

    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                await asyncio.sleep(REQUEST_DELAY)
                r = await client.get(url, timeout=15)

                if r.status_code == 404:
                    return cuit, None, "sin_datos"

                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_retries:
                        wait = (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            "[BCRA] %s → HTTP %d, reintento %d/%d en %.1fs",
                            cuit,
                            r.status_code,
                            attempt + 1,
                            max_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(
                        "[BCRA] %s → HTTP %d tras %d reintentos — abandono",
                        cuit,
                        r.status_code,
                        max_retries,
                    )
                    return cuit, None, "error_fatal"

                r.raise_for_status()
                return cuit, r.json(), "ok"

            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "[BCRA] %s → Timeout, reintento %d/%d en %.1fs",
                        cuit,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "[BCRA] %s → Timeout definitivo tras %d reintentos",
                        cuit,
                        max_retries,
                    )
                    return cuit, None, "error_fatal"

            except Exception as e:
                logger.error("[BCRA] %s → Error inesperado: %s", cuit, e)
                return cuit, None, "error_fatal"

    return cuit, None, "error_fatal"  # no alcanzable, pero satisface mypy


async def _run_padron_async(
    cuits: list[str],
    output_file: Path,
    checkpoint_file: Path,
    max_concurrent: int,
    max_retries: int,
    save_every_n: int,
    log_every_n: int,
) -> None:
    """Corre el pipeline async sobre la lista de CUITs pendientes."""
    total = len(cuits)
    if total == 0:
        logger.info("No hay CUITs nuevos para procesar.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = time.monotonic()
    processed = 0
    con_datos = 0
    errores = 0
    sin_datos = 0

    logger.info(
        "Iniciando: %d CUITs | concurrencia=%d | reintentos=%d | flush cada %d",
        total,
        max_concurrent,
        max_retries,
        save_every_n,
    )

    async with httpx.AsyncClient(headers=HEADERS, verify=False) as client:
        for batch_start in range(0, total, save_every_n):
            batch = cuits[batch_start : batch_start + save_every_n]

            results = await asyncio.gather(
                *[_fetch_one(c, client, semaphore, max_retries) for c in batch]
            )

            batch_rows: list[dict] = []
            batch_cuits: list[str] = []

            for cuit, raw, status in results:
                processed += 1
                batch_cuits.append(cuit)

                if status == "ok" and raw and raw.get("status") == 200:
                    batch_rows.extend(parse_deuda(raw, cuit))
                    con_datos += 1
                elif status == "sin_datos":
                    sin_datos += 1
                else:
                    errores += 1

            # Flush a disco — primero datos, luego checkpoint
            _append_rows_ndjson(output_file, batch_rows)
            _append_checkpoint(checkpoint_file, batch_cuits)

            # Log de progreso cada log_every_n CUITs o al terminar
            if processed % log_every_n == 0 or processed == total:
                elapsed = time.monotonic() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_s = (total - processed) / rate if rate > 0 else float("inf")
                eta_str = f"{eta_s / 60:.1f} min" if eta_s < float("inf") else "?"
                logger.info(
                    "Progreso: %d/%d (%.1f%%) | ✓ con datos: %d | sin datos: %d | errores: %d | ETA: %s",
                    processed,
                    total,
                    processed / total * 100,
                    con_datos,
                    sin_datos,
                    errores,
                    eta_str,
                )

    elapsed_total = time.monotonic() - start_time
    logger.info(
        "Pipeline terminado en %.1f min — %d CUITs procesados, %d errores definitivos",
        elapsed_total / 60,
        processed,
        errores,
    )


def run_pipeline_padron(
    padron_path: Path = PADRON_PATH,
    output_file: Path = OUTPUT_FILE,
    checkpoint_file: Path = CHECKPOINT_FILE,
    max_concurrent: int = MAX_CONCURRENT,
    max_retries: int = MAX_RETRIES,
    save_every_n: int = SAVE_EVERY_N,
    log_every_n: int = LOG_EVERY_N,
    max_cuits: Optional[int] = MAX_CUITS,
) -> None:
    """
    Pipeline de ingesta masiva desde el padrón AFIP.

    Lee CUITs de `padron_path`, descarta los ya procesados según
    `checkpoint_file`, y consulta la Central de Deudores del BCRA
    con concurrencia controlada. Los resultados se guardan
    incrementalmente en `output_file` (NDJSON, una fila por línea).

    Para leer el output:
        df = pd.read_json(output_file, lines=True)

    Para retomar una corrida interrumpida: simplemente volvé a correr
    el script — los CUITs en `checkpoint_file` se saltean automáticamente.

    Args:
        padron_path     : CSV del padrón AFIP (columna 'cuit' requerida)
        output_file     : NDJSON de salida (se crea o se continúa si ya existe)
        checkpoint_file : CUITs ya procesados, uno por línea
        max_concurrent  : requests simultáneos (1 = secuencial, recomendado)
        max_retries     : reintentos ante HTTP 429/5xx con backoff exponencial
        save_every_n    : flush a disco cada N CUITs
        log_every_n     : log de progreso cada N CUITs
        max_cuits       : límite de CUITs a procesar en esta corrida (None = todos)
    """
    df_padron = pd.read_csv(padron_path, dtype={"cuit": str})
    all_cuits = df_padron["cuit"].dropna().unique().tolist()
    logger.info("[Padrón] %d CUITs únicos en %s", len(all_cuits), padron_path.name)

    done = _load_checkpoint(checkpoint_file)
    cuits_nuevos = [c for c in all_cuits if c not in done]
    if max_cuits is not None:
        cuits_nuevos = cuits_nuevos[:max_cuits]
    logger.info(
        "[Checkpoint] %d ya procesados → procesando %d CUITs en esta corrida",
        len(done),
        len(cuits_nuevos),
    )

    asyncio.run(
        _run_padron_async(
            cuits=cuits_nuevos,
            output_file=output_file,
            checkpoint_file=checkpoint_file,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            save_every_n=save_every_n,
            log_every_n=log_every_n,
        )
    )


# ── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────


def run_pipeline(
    n_cuits: int = 50,
    desde_tasas: str = "2023-01-01",
    hasta_tasas: str = "2024-12-31",
    guardar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de descarga.

    Args:
        n_cuits     : cuántos CUITs consultar en Central de Deudores
        desde_tasas : inicio del período para series de tasas
        hasta_tasas : fin del período para series de tasas
        guardar     : si True, guarda parquet/csv localmente

    Returns:
        (df_deudas, df_tasas)
    """
    sep = "=" * 55

    # 1. CUITs desde Argentina Compra
    print(sep)
    print(f"PASO 1 — Obteniendo {n_cuits} CUITs de Argentina Compra")
    print(sep)
    cuits = fetch_cuits_argentina_compra(n_cuits)
    print(f"→ {len(cuits)} CUITs a procesar\n")

    # 2. Central de Deudores
    print(sep)
    print("PASO 2 — Central de Deudores BCRA")
    print(sep)
    rows = []
    sin_datos = 0
    errores = 0

    for i, cuit in enumerate(cuits):
        raw = fetch_deuda_bcra(cuit)

        if raw and raw.get("status") == 200:
            nuevas = parse_deuda(raw, cuit)
            rows.extend(nuevas)
        elif raw is None:
            sin_datos += 1
        else:
            errores += 1

        if (i + 1) % 10 == 0:
            print(
                f"  {i + 1:>3}/{len(cuits)} | filas: {len(rows):>5} | "
                f"sin datos: {sin_datos} | errores: {errores}"
            )

        time.sleep(DELAY_BCRA)

    df_deudas = pd.DataFrame(rows)
    print(f"\n→ df_deudas: {df_deudas.shape}")
    if not df_deudas.empty:
        print(f"  CUITs con datos: {df_deudas['cuit'].nunique()}")
        print(f"  CUITs sin datos: {sin_datos}")

    # 3. Tasas activas
    print(f"\n{sep}")
    print(f"PASO 3 — Tasas activas BCRA ({desde_tasas} → {hasta_tasas})")
    print(sep)
    df_tasas = fetch_tasas_multiples(desde=desde_tasas, hasta=hasta_tasas)
    print(f"→ df_tasas: {df_tasas.shape}")

    # 4. Guardar
    if guardar and not df_deudas.empty:
        df_deudas.to_parquet("deudas_bcra.parquet", index=False)
        print("\n  Guardado: deudas_bcra.parquet")
    if guardar and not df_tasas.empty:
        df_tasas.to_csv("tasas_bcra.csv", index=False)
        print("  Guardado: tasas_bcra.csv")

    print(f"\n{sep}")
    print("Pipeline completo.")
    return df_deudas, df_tasas


# ── DIAGNÓSTICO ──────────────────────────────────────────────────────────────


def describe_deudas(df: pd.DataFrame) -> None:
    """Resumen rápido del DataFrame de deudas para inspección inicial."""
    if df.empty:
        print("DataFrame vacío — puede que ningún CUIT tenga deudas registradas.")
        return

    print(f"Shape            : {df.shape}")
    print(f"CUITs únicos     : {df['cuit'].nunique()}")
    print(f"Períodos         : {sorted(df['periodo'].unique())}")
    print(f"Entidades únicas : {df['entidad'].nunique()}")

    print("\nDistribución de situación crediticia:")
    situ = df.groupby("situacion").agg(
        n_registros=("cuit", "count"),
        cuits_unicos=("cuit", "nunique"),
        monto_total=("monto", "sum"),
    )
    print(situ.to_string())

    print("\nMonto (miles ARS):")
    print(df["monto"].describe().round(1).to_string())

    pct_default = (df["situacion"] >= 3).mean() * 100
    print(f"\n% situación >= 3 (riesgo medio o peor): {pct_default:.1f}%")


def quick_test(cuit: str = "30500010912") -> None:
    """Prueba rápida con un solo CUIT para verificar conectividad."""
    print(f"Testeando CUIT: {cuit}")
    raw = fetch_deuda_bcra(cuit)
    if raw:
        import json

        print(json.dumps(raw, indent=2, ensure_ascii=False)[:2000])
    else:
        print("Sin datos (404 o error de red)")


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # silencia el log de cada request

    # Pipeline masivo desde padrón AFIP (retoma automáticamente si se interrumpe):
    run_pipeline_padron()

    # Alternativas:
    # quick_test()                          # test de conectividad con un CUIT
    # df_deudas, df_tasas = run_pipeline(n_cuits=50)  # pipeline chico vía ArgCompra
    # describe_deudas(df_deudas)
