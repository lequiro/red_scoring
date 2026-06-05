@echo off
REM Setup completo del entorno virtual (Windows)

echo =^> Creando venv con Python 3.11...
py -3.11 -m venv venv

echo =^> Instalando dependencias...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo =^> Generando requirements.txt con versiones exactas...
pip freeze > requirements.txt

echo =^> Creando .vscode/settings.json...
python setup_vscode.py

echo =^> Inicializando git...
git init
git add .
git commit -m "Initial commit: project scaffold"

echo.
echo === Setup completo ===
echo Para activar el venv:
echo   venv\Scripts\activate
