"""
Smoke test del pipeline completo sobre el sample sintético.

Corre graph → communities → features → model → validation redirigiendo TODAS las
rutas a un directorio temporal, y verifica que se generan los artefactos clave.
No toca datos reales ni output/.
"""

from pathlib import Path

import pandas as pd
import pytest

from red_scoring import config, graph, communities, features, model, validation

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
SAMPLE_CSV = SAMPLE_DIR / "bcra_merged_sample.csv"
SAMPLE_PADRON = SAMPLE_DIR / "padron_afip_sample.csv"


@pytest.fixture
def pipeline_dirs(monkeypatch, tmp_path):
    """Redirige todas las rutas de I/O a tmp_path y apunta a los samples."""
    graphs = tmp_path / "graphs"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    for d in (graphs, tables, figures):
        d.mkdir(parents=True, exist_ok=True)

    # entradas
    monkeypatch.setattr(config, "CSV_MERGED", SAMPLE_CSV)
    monkeypatch.setattr(config, "PADRON_MONTO", SAMPLE_PADRON)
    # directorios de salida
    monkeypatch.setattr(config, "GRAPHS_DIR", graphs)
    monkeypatch.setattr(config, "TABLES_DIR", tables)
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    # artefactos (paths derivados, hay que parchearlos uno a uno)
    monkeypatch.setattr(config, "RED_MONTO_GEXF", graphs / "red_monto.gexf")
    monkeypatch.setattr(config, "RED_MONTO_COMUNIDADES_GEXF", graphs / "red_monto_comunidades.gexf")
    monkeypatch.setattr(config, "FEATURES_CSV", tables / "features.csv")
    monkeypatch.setattr(communities, "STATS_CSV", tables / "comunidades_stats.csv")
    monkeypatch.setattr(communities, "HISTOGRAMAS_PNG", figures / "histogramas_comunidades.png")
    monkeypatch.setattr(validation, "CRUCE_CSV", tables / "cruce.csv")
    monkeypatch.setattr(validation, "CRUCE_PNG", figures / "cruce.png")
    monkeypatch.setattr(validation, "ENT_TOP_CSV", tables / "ent_top.csv")
    monkeypatch.setattr(validation, "ENT_MORA_CSV", tables / "ent_mora.csv")
    monkeypatch.setattr(validation, "ENT_PNG", figures / "ent.png")
    return graphs, tables, figures


def test_pipeline_end_to_end(pipeline_dirs):
    graphs, tables, figures = pipeline_dirs

    graph.main()
    assert config.RED_MONTO_GEXF.exists(), "no se generó red_monto.gexf"

    communities.main()
    assert config.RED_MONTO_COMUNIDADES_GEXF.exists(), "no se generó red_monto_comunidades.gexf"

    features.main()
    assert config.FEATURES_CSV.exists(), "no se generó features.csv"

    model.main()
    assert (tables / "scoring_resultados.csv").exists(), "no se generó scoring_resultados.csv"
    assert (figures / "scoring_auc_comparacion.png").exists()

    validation.main_cruce()
    assert validation.CRUCE_CSV.exists()

    validation.main_entidades()
    assert validation.ENT_TOP_CSV.exists()


def test_features_schema(pipeline_dirs):
    graph.main()
    communities.main()
    features.main()

    df = pd.read_csv(config.FEATURES_CSV, index_col=0)
    esperadas = {
        "comportamiento", "es_sano", "log_monto", "comunidad_escala",
        "degree_monto", "pct_vecinos_sanos", "pct_monto_banco", "tiene_fogAr",
    }
    faltantes = esperadas - set(df.columns)
    assert not faltantes, f"faltan columnas en features.csv: {faltantes}"
    # el target es binario y hay ambas clases en el sample
    assert set(df["es_sano"].unique()) == {0, 1}


def test_perfil_comportamiento_clases():
    from red_scoring.io import cargar_datos
    from red_scoring.behavior import perfil_comportamiento

    df = cargar_datos(SAMPLE_CSV, require_monto=False, verbose=False)
    perfil = perfil_comportamiento(df)
    clases = set(perfil["comportamiento"].unique())
    # el sample fue diseñado para cubrir estas clases
    assert {"sano", "cronico", "irrecuperable", "recuperado"} <= clases
