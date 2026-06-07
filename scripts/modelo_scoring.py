#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: modelo de scoring (LOO-CV, AUC) sobre features.csv.
  Salidas: output/figures/scoring_*.png, output/tables/scoring_resultados.csv

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/modelo_scoring.py   (en Spyder: runcell del bloque __main__)
"""

from red_scoring.model import main

if __name__ == "__main__":
    main()
