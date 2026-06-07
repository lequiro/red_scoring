"""Configuración común de tests: backend de matplotlib no-interactivo."""

import matplotlib

matplotlib.use("Agg")
