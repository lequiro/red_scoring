# red_scoring — Scoring crediticio por redes complejas

### Tasas más justas para quienes no tienen historial crediticio

---

## El problema

En Argentina, millones de personas y pymes son **thin-file**: sin historial
crediticio formal, quedan fuera del crédito o pagan tasas de castigo. El scoring
tradicional solo sabe puntuar a quien *ya* tiene historial — un círculo que excluye
precisamente a quien más necesita entrar.

## La solución

`red_scoring` estima el riesgo a partir de la **posición de cada CUIT en una red**,
no solo de su historial individual. Construimos un grafo de **~1.300 CUITs y 33.000
aristas**, donde dos nodos se conectan si comparten entidad financiera y tienen escala
económica similar — todo con **datos 100% públicos** (Central de Deudores del BCRA,
padrón AFIP).

```
BCRA + AFIP  →  grafo de CUITs  →  comunidades  →  features de red  →  score 0–1000
```

## Por qué es creíble: validación honesta

La trampa típica de estos modelos es la **fuga de target** — usar como predictor el
mismo dato que define la etiqueta, lo que infla el AUC a ~1.0 y no prueba nada. La
detectamos y la eliminamos: **ninguna feature de historial entra al modelo**. Por eso
estos números son reales.

| Modelo | Qué "ve" el modelo | AUC-ROC |
|---|---|:---:|
| Baseline | escala + entidad | 0.76 |
| + Red | escala + entidad + red | 0.76 |
| **Solo red** | **solo la posición en la red** | **0.67** |

**Lo relevante para thin-file:** usando *únicamente* la posición en la red —sin
historial ni escala propia— el modelo ordena el riesgo con **AUC 0.67**, muy por
encima del azar (0.50). Es exactamente la población que el sistema tradicional no
puede puntuar.

**Hallazgo accionable:** detectando comunidades por escala, la comunidad de **mayor
escala económica es la más morosa** (~25 % en mora vs. ~7 % en las más chicas) —
contraintuitivo y directamente útil para política de tasas.

## No es un notebook: es un producto

| | |
|---|---|
| **Reproducible** | `make setup && make all` corre el pipeline end-to-end |
| **Empaquetado** | paquete instalable `src/red_scoring/`, un comando por etapa |
| **Sin exponer PII** | datos reales fuera de git + sample sintético para reproducir |
| **Testeado** | `pytest` corre el pipeline completo; **CI verde** en cada push |
| **Stack** | Python 3.11 · NetworkX · scikit-learn · pandas |

## Roadmap

La red de monto es la **primera capa de aristas**. El próximo salto es sumar vínculos
que conecten también a thin-files puros —co-residencia (padrón electoral),
persona–empresa, geografía— donde la hipótesis de *"buenos vecinos, mejor tasa"* tiene
más para dar. La infraestructura para probarlo ya está construida.

## Fuentes de datos

Todo el pipeline se alimenta de **datos públicos**, sin scraping ni datos privados:

- **BCRA — Central de Deudores** (situación crediticia por CUIT, *ground truth*):
  API REST pública `https://api.bcra.gob.ar/CentralDeDeudores/v1.0/Deudas/{cuit}`
- **BCRA — Estadísticas** (entidades financieras y tasas):
  `https://api.bcra.gob.ar/estadisticas/v2.0/` — catálogo de APIs en
  [bcra.gob.ar](https://www.bcra.gob.ar)
- **AFIP — Padrón de contribuyentes** (categoría, actividad y estado por CUIT):
  descarga pública de padrones AFIP

---

<sub>Repositorio: **github.com/lequiro/red_scoring** · Licencia MIT · Datos públicos (BCRA Central de Deudores, BCRA Estadísticas, AFIP)</sub>
