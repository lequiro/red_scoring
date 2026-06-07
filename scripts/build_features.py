#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: ingeniería de features → output/tables/features.csv

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/build_features.py   (en Spyder: runcell del bloque __main__)
"""

from red_scoring.features import main

if __name__ == "__main__":
    main()
