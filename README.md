# Pump It Up: Data Mining the Water Table

Solución completa para la competición [Pump It Up](https://www.drivendata.org/competitions/7/) de DrivenData: predicción del estado funcional de bombas de agua en Tanzania mediante Machine Learning.

**Score final en DrivenData: 0.8203 (top ~25% del leaderboard)**

---

## Descripción del problema

El gobierno tanzano y sus socios necesitan saber qué bombas de agua funcionan correctamente, cuáles necesitan reparación y cuáles están averiadas para priorizar intervenciones de mantenimiento en zonas con escasez hídrica.

El objetivo es predecir el estado de cada bomba (`status_group`) en tres clases:

| Clase | % en train |
|---|---|
| `functional` | 54.31% |
| `non functional` | 38.42% |
| `functional needs repair` | 7.27% |

El dataset contiene **59.400 bombas de entrenamiento** y **14.850 de test**, con 40 variables que incluyen información geográfica, técnica, de gestión y de calidad del agua.

---

## Estructura del repositorio

```
pump-it-up-drivendata/
│
├── driven_data_PumpItUp_Lucia.ipynb   # Notebook principal con todo el análisis
│
├── TrainingSetValues.csv              # Features de entrenamiento (59.400 filas)
├── TrainingSetLabels.csv              # Target de entrenamiento
├── TestSetValues.csv                  # Features de test (14.850 filas)
├── SubmissionFormat.csv               # Formato esperado del submission
│
└── README.md
```

---

## Pipeline del proyecto

### 1. EDA — Análisis Exploratorio
- Informe automático con **YData ProfileReport**: distribuciones, correlaciones, nulos, alertas
- Identificación de **variables redundantes** (grupos de variables que miden lo mismo)
- Detección de **ceros disfrazados** (valores 0 que representan datos desconocidos)

### 2. Preprocesamiento
| Paso | Detalle |
|---|---|
| Eliminación de redundancias | De 41 → 25 variables. Grupos: `quantity/quantity_group`, `source/source_type/source_class`, `extraction_type` x3, etc. |
| Imputación categórica | `funder`, `installer`, `subvillage` → `"unknown"` / `public_meeting`, `permit` → moda del train |
| Imputación numérica | `construction_year`, `longitude`, `latitude`, `population`: ceros → mediana del train |
| Outliers | Mantenidos — Random Forest es robusto y los valores extremos son válidos |
| Normalización tipográfica | `str.strip().str.lower()` en todas las categóricas |
| Agrupación semántica | `pay_monthly` + `pay_annually` → `fixed_payment` en variable `payment` |

### 3. Feature Engineering
| Variable | Descripción |
|---|---|
| `age_pump` | `2013 - construction_year` — antigüedad más informativa que el año absoluto |
| `month_recorded` | Mes de la observación — captura posible estacionalidad |

### 4. Encoding
- **Label Encoding** para todas las categóricas: óptimo para árboles de decisión y evita explosión de columnas con alta cardinalidad (hasta ~19.000 valores únicos en `subvillage`)
- Encoder ajustado exclusivamente sobre train y aplicado al test para evitar data leakage

### 5. Modelado — evolución iterativa

| Versión | Descripción | Score DrivenData |
|---|---|---|
| rf_v1 | Baseline 100 árboles | 0.8096 |
| rf_v3 | + Feature Engineering | 0.8102 |
| rf_v4 | + Feature Selection | 0.8111 |
| rf_v6 | + Profundidad óptima (max_depth=25) | 0.8139 |
| **rf_v7 TOP** | **Pipeline completo + max_depth=20** | **0.8153** |
| **rf_v7 full train** | **Modelo final entrenado con 100% train** | **0.8203** |

### 6. Tuning e iteraciones adicionales exploradas
- **Optuna** para búsqueda de `n_estimators` (50→500) y `max_features` → sin mejora; el cuello de botella estaba en las features
- **Feature Engineering geográfico**: distancia a Dodoma (capital) y Dar es Salaam via fórmula Haversine → alta importancia pero no mejora el score
- **Variables binarias semánticas**: `has_water`, `pays_for_water`, `is_gravity`, etc. → redundantes con las originales
- **Reducción de cardinalidad Top-N + 'rare'**: las variables `subvillage` y `ward` perdían información geográfica valiosa
- **H2O AutoML** (20 min) → 0.7989, peor que RF baseline
- **LightGBM** (500 estimadores) → comparable a RF, sin superarlo

---

## Modelo final

**Random Forest v7** entrenado con el 100% de los datos de train.

```python
RandomForestClassifier(
    n_estimators=175,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
```

**19 variables seleccionadas** (eliminadas por baja importancia o redundancia):
`funder`, `gps_height`, `installer`, `longitude`, `latitude`, `subvillage`, `region`, `district_code`, `lga`, `ward`, `population`, `extraction_type`, `management`, `payment`, `quantity`, `source`, `waterpoint_type`, `age_pump`, `month_recorded`

**Variables más importantes:**
`quantity` (0.12) > `longitude` (0.11) > `latitude` (0.10) > `subvillage` (0.07) > `waterpoint_type` (0.06) > `gps_height` (0.06)

**Resultados en validación (80/20 split):**
```
                         precision    recall  f1-score
functional                  0.80      0.91      0.85
functional needs repair     0.62      0.27      0.38
non functional              0.86      0.77      0.81
accuracy                                        0.81
```

---

## Conclusiones

**Lo que funcionó:**
- Feature Engineering con `age_pump` en lugar del año absoluto
- Eliminar variables de baja importancia mejora la generalización (menos ruido > más features)
- Limitar `max_depth=20` evita overfitting — con `max_depth=25` el score local era mayor pero el real menor
- Normalización tipográfica para unificar categorías con distintas grafías

**Lo que no funcionó:**
- Complejidad adicional: variables binarias, distancias, reducción de cardinalidad
- Optuna no pudo mejorar el modelo — el techo estaba en las features, no en los hiperparámetros
- H2O AutoML y LightGBM no superaron al Random Forest

**Aprendizaje clave:** El modelo más sencillo fue el mejor. Más variables y más complejidad no equivalen a mejor generalización. La clave está en un preprocesamiento cuidadoso y eliminar el ruido.

---

## Tecnologías

| Categoría | Librería |
|---|---|
| Manipulación de datos | `pandas`, `numpy` |
| Visualización | `matplotlib`, `seaborn` |
| Preprocesamiento | `scikit-learn` |
| EDA automático | `ydata-profiling` |
| Modelado | `scikit-learn` (RandomForest), `LightGBM` |
| AutoML | `H2O` |
| Optimización de hiperparámetros | `Optuna` |

---

## Cómo reproducir

```bash
# 1. Clonar el repositorio
git clone https://github.com/luciaherranz/pump-it-up-drivendata.git
cd pump-it-up-drivendata

# 2. Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling optuna lightgbm h2o imbalanced-learn

# 3. Abrir el notebook
jupyter notebook driven_data_PumpItUp_Lucia.ipynb
```

Los datos ya están incluidos en el repositorio. Si prefieres descargarlos directamente, están disponibles en la [página de la competición en DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/).

---

## Autora

**Lucía Herranz Somoza**  
[LinkedIn](https://www.linkedin.com/in/lucia-herranz-somoza) · [GitHub](https://github.com/luciaherranz)
