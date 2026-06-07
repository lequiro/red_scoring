#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point: detección de comunidades sobre red_monto.
  Salidas: output/graphs/red_monto_comunidades.gexf, tables/, figures/

Requiere `pip install -e .` (o `make setup`) una vez. Luego:
  python scripts/analisis_comunidades.py   (en Spyder: runcell del bloque __main__)
"""

from red_scoring.communities import main

if __name__ == "__main__":
    main()
