#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de scoring predictivo.

  Entrada : output/tables/features.csv   (producido por build_features.py)

  ⚠ Las features de historial [A] (derivadas de `situacion`) NO se usan: definen
    el target es_sano y producirían fuga (AUC≈1 trivial). Ver MODELOS abajo.

  Experimentos (ninguno usa [A]; red_morosos quedó fuera del pipeline):
    M0 — Baseline   : escala [B] + entidad [D]            (info sin historial)
    M1 — Red completa: [B] + vecindad red_monto [C] + entidad [D]
    M2 — Solo red    : vecindad red_monto [C]             (thin-file benchmark)

  Validación:
    Leave-One-Out CV sobre los nodos etiquetados.
    Métrica principal: AUC-ROC (más informativa que accuracy con clases desbalanceadas).

  Salida:
    output/figures/scoring_auc_comparacion.png    — barras AUC por modelo
    output/figures/scoring_feature_importance.png — importancia (Random Forest)
    output/figures/scoring_scores_distribucion.png — distribución del score por clase
    output/tables/scoring_resultados.csv          — scores individuales por CUIT

Correr: python3 modelo_scoring.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

AQUI        = os.path.dirname(os.path.abspath(__file__))
# Esquema de salidas: output/{graphs,tables,figures}
OUT_DIR     = os.path.normpath(os.path.join(AQUI, "..", "output"))
TABLES_DIR  = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
CSV_IN      = os.path.join(TABLES_DIR, "features.csv")

# ── grupos de features ──────────────────────────────────────────────────────────

# ⚠ FUGA DE TARGET — el grupo [A] se deriva de la misma serie de `situacion`
# que define es_sano (sano ⟺ peor_situacion==1). Si entra como predictor, el
# modelo consigue AUC≈1 trivialmente y la comparación "red vs no-red" pierde
# sentido. Por eso NO se usa en ningún modelo; queda listado solo como
# referencia de qué columnas hay que evitar.
FEATS_A_LEAKY = [   # historial individual — NO usar como predictor (fuga de target)
    "peor_situacion", "situacion_final", "n_periodos_mora",
    "racha_max_mora", "tendencia", "n_transiciones", "volatilidad",
]
FEATS_B = [   # escala económica
    "log_monto", "n_periodos", "comunidad_escala",
]
FEATS_C = [   # vecindad en red_monto (co-escala)
    "degree_monto", "pct_vecinos_sanos", "media_pond_sit_vecinos",
]
FEATS_D = [   # entidad financiera
    "pct_monto_banco", "tiene_fogAr", "n_entidades_distintas",
]
# (red_morosos quedó fuera del pipeline: ya no hay grupo [E].)

# Ningún modelo usa [A]: el baseline honesto para thin-file es la info
# disponible SIN mirar el historial que define la etiqueta (escala + entidad),
# y se mide cuánto agrega la red de monto por encima de eso.
MODELOS = {
    "M0 Baseline (escala+entidad)":  FEATS_B + FEATS_D,
    "M1 Red completa":               FEATS_B + FEATS_C + FEATS_D,
    "M2 Solo red (thin-file)":       FEATS_C,
}


# ── helpers ─────────────────────────────────────────────────────────────────────

def hacer_pipeline():
    """LogisticRegression con imputación de medianas y normalización."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])


def loo_auc(X, y):
    """Leave-One-Out AUC-ROC. Devuelve AUC y array de probabilidades predichas."""
    loo = LeaveOneOut()
    scores = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        pipe = hacer_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        scores[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    auc = roc_auc_score(y, scores)
    return auc, scores


def rf_importancia(X, y, feature_names):
    """Random Forest entrenado sobre todo el set — importancias de Gini."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_imp, y)
    imp_df = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    return imp_df


# ── plots ────────────────────────────────────────────────────────────────────────

def plot_auc_comparacion(resultados):
    fig, ax = plt.subplots(figsize=(8, 4))
    nombres = list(resultados.keys())
    aucs    = [resultados[n]["auc"] for n in nombres]
    # un color por modelo, robusto a cualquier cantidad de modelos
    cmap    = plt.get_cmap("tab10")
    colores = [cmap(i % 10) for i in range(len(nombres))]
    bars = ax.barh(nombres, aucs, color=colores, edgecolor="white", height=0.5)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="azar")
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("AUC-ROC (LOO-CV)")
    ax.set_title("Comparación de modelos — AUC-ROC")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "scoring_auc_comparacion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"PNG: {path}")


def plot_feature_importance(imp_df):
    top = imp_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#2171b5", edgecolor="white")
    ax.set_xlabel("Importancia Gini (Random Forest)")
    ax.set_title("Top-15 features — modelo M1 completo")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "scoring_feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"PNG: {path}")


def plot_score_distribucion(df_scores, nombre_modelo):
    """Distribución del score M1 por clase de comportamiento."""
    clases = sorted(df_scores["comportamiento"].unique())
    colores = {"sano": "#2ca02c", "recuperado": "#1f77b4",
               "cronico": "#ff7f0e", "irrecuperable": "#d62728",
               "sin_historial": "#9467bd"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # histograma superpuesto
    ax = axes[0]
    for cls in clases:
        vals = df_scores.loc[df_scores["comportamiento"] == cls, "score_m1"]
        ax.hist(vals, bins=20, alpha=0.6, label=f"{cls} (n={len(vals)})",
                color=colores.get(cls, "grey"), density=True)
    ax.set_xlabel("Score (P(sano))")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del score por comportamiento")
    ax.legend(fontsize=8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)

    # box plots
    ax = axes[1]
    data_box = [df_scores.loc[df_scores["comportamiento"] == cls, "score_m1"].values
                for cls in clases]
    bp = ax.boxplot(data_box, patch_artist=True, notch=False)
    for patch, cls in zip(bp["boxes"], clases):
        patch.set_facecolor(colores.get(cls, "grey"))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(clases) + 1))
    ax.set_xticklabels(clases, rotation=20, ha="right")
    ax.set_ylabel("Score (P(sano))")
    ax.set_title("Boxplot score por comportamiento")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Modelo: {nombre_modelo}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "scoring_scores_distribucion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"PNG: {path}")


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── carga ──────────────────────────────────────────────────────────────────
    print(f"Cargando {CSV_IN} ...")
    df = pd.read_csv(CSV_IN, index_col=0)
    print(f"Shape: {df.shape}")

    y = df["es_sano"].to_numpy()
    print(f"\nDistribución target — sano: {y.sum()} / total: {len(y)} ({100*y.mean():.1f}%)")

    # ── LOO-CV por modelo ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Leave-One-Out Cross-Validation")
    print("="*60)

    resultados = {}
    for nombre, feats in MODELOS.items():
        feats_presentes = [f for f in feats if f in df.columns]
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  [WARN] {nombre}: features ausentes {missing}")

        X = df[feats_presentes].to_numpy()
        auc, scores = loo_auc(X, y)
        resultados[nombre] = {"auc": auc, "scores": scores, "feats": feats_presentes}
        print(f"  {nombre:35s}  AUC = {auc:.4f}")

    # ── feature importance M1 ─────────────────────────────────────────────────
    print("\nCalculando importancia de features (Random Forest, M1) ...")
    feats_m1 = [f for f in MODELOS["M1 Red completa"] if f in df.columns]
    X_m1 = df[feats_m1].to_numpy()
    imp_df = rf_importancia(X_m1, y, feats_m1)
    print("\nTop-10 features:")
    print(imp_df.head(10).to_string())

    # ── guardar scores ────────────────────────────────────────────────────────
    df_scores = df[["comportamiento", "es_sano"]].copy()
    df_scores["score_m0"] = resultados["M0 Baseline (escala+entidad)"]["scores"]
    df_scores["score_m1"] = resultados["M1 Red completa"]["scores"]
    df_scores["score_m2"] = resultados["M2 Solo red (thin-file)"]["scores"]

    # score en escala 0-1000 (más familiar para presentación)
    df_scores["score_m1_1000"] = (df_scores["score_m1"] * 1000).round().astype(int)

    path_csv = os.path.join(TABLES_DIR, "scoring_resultados.csv")
    df_scores.to_csv(path_csv)
    print(f"\nCSV con scores: {path_csv}")

    # ── estadísticas por clase ────────────────────────────────────────────────
    print("\nScore M1 por clase de comportamiento:")
    stats = df_scores.groupby("comportamiento")["score_m1"].agg(
        n="count", media="mean", mediana="median", std="std",
        p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75)
    ).round(3)
    print(stats.to_string())

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerando gráficos ...")
    plot_auc_comparacion(resultados)
    plot_feature_importance(imp_df)
    plot_score_distribucion(df_scores, "M1 Red completa")

    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    aucs = {k: v["auc"] for k, v in resultados.items()}
    delta_m1 = aucs["M1 Red completa"] - aucs["M0 Baseline (escala+entidad)"]
    print(f"  Baseline (M0)        : {aucs['M0 Baseline (escala+entidad)']:.4f}")
    print(f"  Red completa (M1)    : {aucs['M1 Red completa']:.4f}  (Δ vs M0 = {delta_m1:+.4f})")
    print(f"  Solo red (M2)        : {aucs['M2 Solo red (thin-file)']:.4f}")
    print()
    if delta_m1 > 0.01:
        print("  ✓ La red de monto aporta señal real por encima de escala+entidad (+AUC > 0.01)")
    elif delta_m1 > 0:
        print("  ~ La red de monto aporta señal marginal sobre escala+entidad")
    else:
        print("  ✗ La red de monto no mejora el baseline de escala+entidad")
    print(f"  M2 (solo vecindad de red) sobre azar (0.5): {aucs['M2 Solo red (thin-file)'] - 0.5:+.4f}"
          " — referencia para perfiles thin-file sin escala/entidad propias")


if __name__ == "__main__":
    main()
