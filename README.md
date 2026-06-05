# red_scoring

Credit scoring alternativo basado en análisis de redes complejas para perfiles sin historial crediticio formal (thin-file), utilizando datos públicos argentinos.

## Installation

```bash
python3.11 -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Copiar `.env.example` a `.env` y completar las variables:

```bash
cp .env.example .env
```

## Usage

```bash
python main.py
```

## Project Structure

```
red_scoring/
├── src/                  # Código fuente principal
├── tests/                # Pruebas unitarias
├── data/                 # Datos locales (ignorado por git)
├── notebooks/            # Exploración y prototipado
├── main.py               # Entrada principal
├── config.py             # Carga de variables de entorno
├── requirements.txt      # Dependencias con versiones fijas
├── .env.example          # Plantilla de variables de entorno
├── .gitignore
└── README.md
```
