#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: cruza comunidad de escala (red_monto) x comportamiento.
  Salidas: output/tables/cruce_comunidad_comportamiento.csv, figures/cruce_heatmap.png

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/validacion_cruce.py
"""

from red_scoring.validation import main_cruce

if __name__ == "__main__":
    main_cruce()
