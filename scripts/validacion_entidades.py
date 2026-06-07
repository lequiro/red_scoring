#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: validación por entidad financiera (control de sesgo por banco).
  Salidas: output/tables/validacion_entidades_*.csv, figures/validacion_entidades_heatmap.png

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/validacion_entidades.py
"""

from red_scoring.validation import main_entidades

if __name__ == "__main__":
    main_entidades()
