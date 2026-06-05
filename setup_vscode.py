"""
Ejecutar una vez para crear .vscode/settings.json
  python setup_vscode.py
"""
import json, pathlib

settings = {
    "python.defaultInterpreterPath": "./venv/bin/python",
    "editor.formatOnSave": True,
    "python.formatting.provider": "black",
    "python.linting.enabled": True,
    "python.linting.flake8Enabled": True,
}

vscode_dir = pathlib.Path(".vscode")
vscode_dir.mkdir(exist_ok=True)
(vscode_dir / "settings.json").write_text(json.dumps(settings, indent=4))
print(".vscode/settings.json creado correctamente.")
