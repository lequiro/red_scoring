"""
config.py — carga de variables de entorno desde .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
