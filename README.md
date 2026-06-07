# red_scoring

Credit scoring alternativo basado en **análisis de redes complejas** para perfiles
sin historial crediticio formal (*thin-file*), usando datos públicos argentinos.

[![CI](https://github.com/lequiro/red_scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/lequiro/red_scoring/actions/workflows/ci.yml)

## Idea

> Hipótesis: los vecinos crediticiamente sanos de un nodo en la red reducen su
> probabilidad de default, más allá de su historial individual — lo que permitiría
> asignar mejores tasas a perfiles thin-file bien conectados.

Se construye una red de CUITs (la **red de monto**, `red_monto`) donde dos nodos se
conectan si comparten una entidad financiera y tienen una escala económica similar,
se detectan comunidades, se derivan features de red y se evalúa con un modelo si la
topología aporta señal por encima de los features individuales.

El *ground truth* es la situación crediticia de la **Central de Deudores del BCRA**.

## Arquitectura del pipeline

```
data/bcra_merged.csv
        │
        ▼
 build_red_monto      → output/graphs/red_monto.gexf            (nodos=CUITs, aristas=escala+entidad común)
        │
        ▼
 analisis_comunidades → output/graphs/red_monto_comunidades.gexf (+ stats, histogramas)
        │
        ▼
 build_features       → output/tables/features.csv              (features [B] escala, [C] vecindad, [D] entidad)
        │
        ▼
 modelo_scoring       → AUC por modelo + scores                 (LOO-CV; M0 baseline vs M1 con red)
        │
        ▼
 validacion_*         → cruces comunidad×comportamiento, sesgo por entidad
```

Cada etapa es un entry point fino en `scripts/`; la lógica vive en el paquete
`src/red_scoring/` (`io`, `graph`, `behavior`, `communities`, `features`, `model`,
`validation`, `config`).

## Instalación

Requiere Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
make setup                      # = pip install -e ".[dev]"
```

`make setup` instala el paquete en modo editable; tras eso `import red_scoring`
funciona en CLI y en Spyder.

## Datos

Los datasets reales **no se versionan** (pesan y contienen datos personales: CUIT y
denominación). Ver [`data/README.md`](data/README.md) para el schema, las fuentes y
la fecha de corte. Para regenerarlos desde las APIs públicas:

```bash
make data        # bcra_fetch.py (ingesta BCRA) + merge_bcra.py
```

El repo incluye un sample sintético en `data/sample/` para correr el pipeline y los
tests sin datos reales.

## Uso

```bash
make all         # graph → communities → features → model → validate
```

Etapas sueltas: `make graph`, `make communities`, `make features`, `make model`,
`make validate`. `make clean` borra `output/`.

## Salidas (`output/`)

| Carpeta | Contenido |
|---|---|
| `graphs/` | `.gexf` de la red y la red con comunidades (abrir en Gephi) |
| `tables/` | `features.csv`, `scoring_resultados.csv`, stats y cruces |
| `figures/` | comparación de AUC, importancia de features, heatmaps |

## Resultados (con los datos actuales)

Sin fuga de target (las features de historial que *definen* la etiqueta se excluyen):

| Modelo | Features | AUC-ROC (LOO) |
|---|---|---|
| M0 Baseline | escala + entidad | 0.764 |
| M1 Red completa | escala + entidad + vecindad de red | 0.764 |
| M2 Solo red | vecindad de red | 0.667 |

**Lectura:** la red de monto sola supera al azar (0.667), pero **no aporta señal
marginal** por encima de escala + entidad (M1 ≈ M0). La hipótesis no se confirma para
esta capa de aristas; conectar thin-files probablemente requiere otra fuente de
vínculos (co-residencia, persona–empresa, etc.). El experimento `red_morosos` vive en
la rama `experiment/red-morosos`.

## Limitaciones

- ~1/3 de la economía argentina es informal y no aparece en registros formales.
- Los CUITs sin deuda activa quedan fuera de `red_monto` (no tienen escala medible).
- AUC sobre predicciones Leave-One-Out es indicativo; con más datos conviene k-fold
  estratificado con intervalos de confianza.
- Datos personales: no subir `data/` real a ningún lado.

## Desarrollo

```bash
pytest -q              # corre el pipeline sobre el sample
flake8 src tests
black src tests scripts
```

## Licencia

MIT — ver [LICENSE](LICENSE).
