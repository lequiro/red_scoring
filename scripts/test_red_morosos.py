#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacion de build_red_morosos.py sobre un caso chico (4 CUITs).
Correr:  python3 scripts/test_red_morosos.py
Si todo da bien, imprime "VALIDACION OK".
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_red_morosos import perfil_comportamiento, construir_grafo

# cuit, periodo, entidad, monto, situacion  (peor situacion por periodo entre [])
filas = [
    # CUIT 11 -> [3,3,3] cronico ; entidades Banco1, Banco2
    ("11", 202401, "Banco1", 100, 3), ("11", 202401, "Banco2", 50, 1),
    ("11", 202402, "Banco1", 100, 3), ("11", 202402, "Banco2", 50, 1),
    ("11", 202403, "Banco1", 100, 3), ("11", 202403, "Banco2", 50, 1),
    # CUIT 22 -> [2,3,3] cronico ; entidades Banco1, Banco3
    ("22", 202401, "Banco1", 80, 2), ("22", 202401, "Banco3", 10, 1),
    ("22", 202402, "Banco1", 80, 3), ("22", 202402, "Banco3", 10, 1),
    ("22", 202403, "Banco1", 80, 3), ("22", 202403, "Banco3", 10, 1),
    # CUIT 33 -> [1,1,0] sano ; entidad Banco1
    ("33", 202401, "Banco1", 200, 1),
    ("33", 202402, "Banco1", 200, 1),
    ("33", 202403, "Banco1", 200, 0),
    # CUIT 44 -> [3,3,0] salida ; entidad Banco1
    ("44", 202401, "Banco1", 300, 3),
    ("44", 202402, "Banco1", 300, 3),
    ("44", 202403, "Banco1", 0, 0),
]
df = pd.DataFrame(filas, columns=["cuit", "periodo", "entidad", "monto", "situacion"])

perfil = perfil_comportamiento(df)
ent_map = df[df["monto"] > 0].groupby("cuit")["entidad"].apply(set).to_dict()
G = construir_grafo(perfil, ent_map)

# --- Clases esperadas ---
assert perfil.loc["11", "comportamiento"] == "cronico", perfil.loc["11"].to_dict()
assert perfil.loc["22", "comportamiento"] == "cronico", perfil.loc["22"].to_dict()
assert perfil.loc["33", "comportamiento"] == "sano", perfil.loc["33"].to_dict()
assert perfil.loc["44", "comportamiento"] == "salida", perfil.loc["44"].to_dict()

# --- Features puntuales ---
assert perfil.loc["11", "racha_max_mora"] == 3
assert perfil.loc["11", "n_periodos_mora"] == 3
assert perfil.loc["44", "situacion_final"] == 0
assert perfil.loc["44", "racha_max_mora"] == 2

# --- Estructura de la red ---
assert G.number_of_nodes() == 4, G.number_of_nodes()
assert G.number_of_edges() == 1, list(G.edges(data=True))
assert G.has_edge("11", "22"), "deberia conectar a los dos cronicos"
assert abs(G["11"]["22"]["weight"] - 1.0) < 1e-9, G["11"]["22"]
assert G["11"]["22"]["n_entidades"] == 1, G["11"]["22"]
assert G.degree("33") == 0 and G.degree("44") == 0, "sano/salida deben quedar aislados"

print("\nNodos:", dict(G.nodes(data="comportamiento")))
print("Aristas:", list(G.edges(data=True)))
print("\nVALIDACION OK")
