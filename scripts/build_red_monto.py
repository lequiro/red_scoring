#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: construye la red de escala económica (red_monto).
  Salida: output/graphs/red_monto.gexf

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/build_red_monto.py   (en Spyder: runcell del bloque __main__)
"""

from red_scoring.graph import main

if __name__ == "__main__":
    main()
