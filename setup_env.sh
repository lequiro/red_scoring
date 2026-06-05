#!/usr/bin/env bash
# Setup completo del entorno virtual (Linux/Mac)
set -e

echo "==> Creando venv con Python 3.11..."
python3.11 -m venv venv

echo "==> Activando venv..."
source venv/bin/activate

echo "==> Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Generando requirements.txt con versiones exactas..."
pip freeze > requirements.txt

echo "==> Creando .vscode/settings.json..."
python setup_vscode.py

echo "==> Inicializando git..."
git init
git add .
git commit -m "Initial commit: project scaffold"

echo ""
echo "=== Setup completo ==="
echo "Para activar el venv:"
echo "  source venv/bin/activate"
