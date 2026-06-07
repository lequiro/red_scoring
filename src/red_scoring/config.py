"""
Configuración central del pipeline: rutas, constantes y semillas.

Todas las rutas se derivan de la ubicación de este archivo
(src/red_scoring/config.py), así que el pipeline funciona sin importar desde
qué directorio se ejecute (CLI, Spyder runcell, tests).
"""

from pathlib import Path

# ── rutas base ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# salidas organizadas por tipo
GRAPHS_DIR = OUTPUT_DIR / "graphs"     # .gexf
TABLES_DIR = OUTPUT_DIR / "tables"     # .csv
FIGURES_DIR = OUTPUT_DIR / "figures"   # .png

# ── datasets de entrada ──────────────────────────────────────────────────────

CSV_MERGED = DATA_DIR / "bcra_merged.csv"
PADRON_MONTO = DATA_DIR / "padron_afip_1.csv"

# ── artefactos del pipeline ──────────────────────────────────────────────────

RED_MONTO_GEXF = GRAPHS_DIR / "red_monto.gexf"
RED_MONTO_COMUNIDADES_GEXF = GRAPHS_DIR / "red_monto_comunidades.gexf"
FEATURES_CSV = TABLES_DIR / "features.csv"

# ── parámetros del modelo de red ─────────────────────────────────────────────

# Umbral de similitud de escala en espacio log-natural.
# OJO: 0.5 NO es "1 orden de magnitud". En log natural, |Δ|=0.5 equivale a un
# factor e^0.5 ≈ 1.65x entre los montos; un orden de magnitud sería ln(10) ≈ 2.30.
DELTA_MAX = 0.5

# semilla única para reproducibilidad (modelos, cualquier muestreo)
SEED = 42

# columnas del padrón AFIP que se adjuntan como atributos de nodo
AFIP_COLS = [
    "denominacion",
    "estado_afip",
    "condicion_iva",
    "cat_ganancias",
    "empleador",
    "monotributo",
    "act_monotributo",
]


def asegurar_dirs(*dirs: Path) -> None:
    """Crea los directorios de salida que se le pasen (idempotente)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
