"""
Clasificación de comportamiento crediticio por CUIT a partir de la trayectoria
de `situacion`, y features de historial [A].

Es la ÚNICA fuente de `comportamiento` en el pipeline (red_morosos quedó fuera).

⚠ FUGA DE TARGET: las features [A] (peor_situacion, situacion_final, ...) se
derivan de la misma serie de `situacion` que define `comportamiento`/es_sano
(sano ⟺ peor_situacion==1). Sirven como descriptor/EDA, pero NO deben usarse
como predictores: modelo_scoring.py las excluye.
"""

import numpy as np
import pandas as pd


def racha_max(mask) -> int:
    """Racha más larga de True consecutivos (vectorizado)."""
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def perfil_comportamiento(df) -> pd.DataFrame:
    """
    Etiqueta + features [A] por CUIT, sobre la peor situacion de cada periodo.

    Clases:
      sin_historial : nunca tuvo deuda activa (rama defensiva; situacion BCRA ≥ 1)
      sano          : siempre al corriente (peor situacion == 1)
      irrecuperable : tocó situacion 5/6, o terminó en 4+
      recuperado    : tuvo mora pero normalizó (termina en 1)
      cronico       : sigue en mora
    """
    peor = df.groupby(["cuit", "periodo"])["situacion"].max().sort_index()
    filas = {}
    for cuit, s in peor.groupby(level=0):
        a = s.to_numpy().astype(int)
        maxs, final = int(a.max()), int(a[-1])
        if maxs == 0:
            clase = "sin_historial"
        elif maxs == 1:
            clase = "sano"
        elif (a >= 5).any() or final >= 4:
            # situacion 5 (irrecuperable) y 6 (irrec. por disp. técnica)
            clase = "irrecuperable"
        elif final <= 1:
            clase = "recuperado"
        else:
            clase = "cronico"
        filas[cuit] = {
            "comportamiento": clase,
            "peor_situacion":  maxs,
            "situacion_final": final,
            "n_periodos_mora": int((a >= 2).sum()),
            "racha_max_mora":  racha_max(a >= 2),
            "tendencia":       round(float(np.polyfit(np.arange(a.size), a, 1)[0]), 4)
                               if a.size > 1 else 0.0,
            "n_transiciones":  int((np.diff(a) != 0).sum()),
            "volatilidad":     round(float(a.std()), 4),
        }
    perfil = pd.DataFrame.from_dict(filas, orient="index")
    print("Clases:", perfil["comportamiento"].value_counts().to_dict())
    return perfil
