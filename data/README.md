# Datos

Esta carpeta está **gitignoreada**: los datasets reales NO se versionan (pesan y
contienen datos personales). Lo único versionado es este README y `sample/`.

El pipeline regenera los datos desde fuentes públicas, así que no hace falta
distribuir los archivos: se reconstruyen con `make data` (ver más abajo).

> ⚠ **Datos personales.** `cuit` y `denominacion` identifican personas/empresas
> reales. No los subas a ningún repo ni los compartas. Para demos y tests usá
> `sample/` (100% sintético).

---

## Archivos

| Archivo | Qué es | Cómo se obtiene |
|---|---|---|
| `bcra_merged.csv` | Central de Deudores del BCRA unificada (entrada del pipeline) | `make data` → `bcra_fetch.py` + `merge_bcra.py` |
| `padron_afip_1.csv` | Padrón AFIP (atributos de nodo para red_monto) | Descarga bulk de AFIP (ver abajo) |
| `padron_afip.csv` | Padrón AFIP (variante usada por la rama red_morosos) | Ídem |
| `sample/bcra_merged_sample.csv` | Sample sintético de `bcra_merged.csv` (para tests/demo) | Versionado en el repo |
| `sample/padron_afip_sample.csv` | Sample sintético del padrón | Versionado en el repo |

**Fecha de corte de los datos actuales:** 2026-06-07 (columna `fecha_consulta`).

---

## Schema de `bcra_merged.csv`

Una fila = un registro `(cuit, periodo, entidad)` de la Central de Deudores.

| Columna | Tipo | Descripción |
|---|---|---|
| `cuit` | str (11 díg.) | Identificador del deudor. **PII.** |
| `denominacion` | str | Nombre/razón social. **PII.** Puede estar vacío. |
| `periodo` | str `AAAAMM` | Período informado (ej. `202412`). Vacío en filas `sin_datos`. |
| `entidad` | str | Nombre de la entidad financiera (ej. `BANCO MACRO S.A.`). |
| `monto_miles_pesos` | float | Deuda en miles de pesos. |
| `situacion` | float `1..6` | Situación crediticia BCRA: 1 normal … 5/6 irrecuperable. `0` = sin deuda activa en el período. |
| `refinanciaciones` | float `0/1` | Flag. |
| `situacion_juridica` | float `0/1` | Flag. |
| `proceso_judicial` | float `0/1` | Flag. |
| `en_revision` | float `0/1` | Flag. |
| `dias_atraso` | float | Días de atraso. |
| `recategorizacion_oblig` | float `0/1` | Flag (puede ser NaN: no está en los CSV originales). |
| `tipo_registro` | str | `historial` (tiene deuda) o `sin_datos` (CUIT sin registro). |
| `fecha_consulta` | date | Fecha en que se consultó la API. |

> Nota: el pipeline parsea `periodo` de forma robusta (acepta `AAAAMM` o `AAAA-MM`).
> Las filas `sin_datos` (sin `entidad`) se descartan en la carga.

## Schema del padrón AFIP

`cuit` + columnas de atributo: `denominacion`, `estado_afip`, `condicion_iva`,
`cat_ganancias`, `empleador`, `monotributo`, `act_monotributo`. Se adjuntan como
atributos de nodo en `red_monto` (ver `red_scoring.config.AFIP_COLS`).

---

## Regenerar los datos

```bash
make data        # bcra_fetch.py (ingesta BCRA + Argentina Compra) + merge_bcra.py
```

`bcra_fetch.py` consulta la API pública del BCRA con checkpoint reanudable;
`merge_bcra.py` unifica los CSV/NDJSON en `bcra_merged.csv`.

El **padrón AFIP** no se baja por API: hay que descargar el bulk de AFIP y dejar
`padron_afip_1.csv` en esta carpeta. Si el archivo falta, el pipeline corre igual
(los atributos AFIP quedan vacíos).

## Correr el pipeline sobre el sample (sin datos reales)

Los tests (`tests/`) usan `sample/`. Para una corrida manual, apuntá las rutas de
`red_scoring.config` al sample, o copiá `sample/bcra_merged_sample.csv` a
`data/bcra_merged.csv`.
