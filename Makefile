# Pipeline red_scoring — orquestación de un comando.
#
# Uso:
#   make setup       # instala el paquete (editable) + deps
#   make all         # corre el pipeline completo (asume data/ ya generado)
#   make data        # (opcional) regenera data/ desde las APIs públicas
#   make clean       # borra las salidas de output/
#
# Etapas individuales: graph, communities, features, model, validate

PY := python

.PHONY: setup data graph communities features model validate all clean

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

# Ingesta desde fuentes públicas (BCRA, Argentina Compra). Requiere red.
data:
	$(PY) scripts/bcra_fetch.py
	$(PY) scripts/merge_bcra.py

graph:
	$(PY) scripts/build_red_monto.py

communities:
	$(PY) scripts/analisis_comunidades.py

features:
	$(PY) scripts/build_features.py

model:
	$(PY) scripts/modelo_scoring.py

validate:
	$(PY) scripts/validacion_cruce.py
	$(PY) scripts/validacion_entidades.py

# Pipeline completo (no incluye `data`: se asume bcra_merged.csv presente)
all: graph communities features model validate

clean:
	rm -rf output/graphs/* output/tables/* output/figures/*
