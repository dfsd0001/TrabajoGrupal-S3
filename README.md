# Modelos No Supervisados — AI4I Predictive Maintenance

## Segmentación de Perfiles Operacionales de Maquinaria Industrial

**Materia:** Aprendizaje Automático  
**Integrantes:**

| Nombre | Rol |
|--------|-----|
| Madheline Katerine Torres Hallo | Miembro del equipo |
| Carlos Vladimir Ramírez Espinoza | Miembro del equipo |
| Dennys Francisco Salazar Domínguez | Miembro del equipo |

---

## Descripción del Proyecto

Este proyecto implementa y compara algoritmos de aprendizaje no supervisado sobre el dataset **AI4I 2020 Predictive Maintenance** con el objetivo de identificar perfiles operacionales de maquinaria industrial, detectar anomalías y comunicar hallazgos de forma técnica y visual.

El contexto del caso simula una plataforma industrial de monitoreo de activos que necesita segmentar los perfiles de operación de sus máquinas para optimizar estrategias de mantenimiento predictivo, reducir tiempos no planificados de parada y adaptar los umbrales de alerta por régimen operacional.

---

## Dataset

**Nombre:** AI4I 2020 Predictive Maintenance Dataset  
**Fuente:** UCI Machine Learning Repository  
**Registros:** 10 000  
**Variables:** 8 (5 variables de sensores + tipo de máquina + identificador + variable de fallo)

| Variable | Tipo | Descripción |
|---|---|---|
| `UDI` | Entero | Identificador único de registro (excluido del análisis) |
| `Type` | Categórica | Tipo de máquina: L (bajo), M (medio), H (alto) |
| `Air temperature [K]` | Continua | Temperatura del aire en Kelvin |
| `Process temperature [K]` | Continua | Temperatura del proceso en Kelvin |
| `Rotational speed [rpm]` | Continua | Velocidad rotacional en RPM |
| `Torque [Nm]` | Continua | Torque en Newton-metro |
| `Tool wear [min]` | Continua | Desgaste acumulado de herramienta en minutos |
| `Machine failure` | Binaria | Variable de fallo: 0 = sin fallo, 1 = fallo (referencia, no usada en clustering) |

**Distribución de fallos:** 9 664 registros sin fallo (96.6%) — 336 con fallo (3.4%)

---

## Estructura del Repositorio

```
.
├── assets/                          # Visualizaciones exportadas
│   ├── 01_distribucion_variables.png
│   ├── 02_boxplots_tipo_maquina.png
│   ├── 03_matriz_correlacion.png
│   ├── 04_pairplot.png
│   ├── 05_seleccion_k.png
│   ├── 06_kmeans_clusters.png
│   ├── 07_kdistance_graph.png
│   ├── 08_dbscan_clusters.png
│   ├── 09_pca_visualizacion.png
│   ├── 10_tsne_visualizacion.png
│   ├── 11_deteccion_anomalias.png
│   ├── 12_comparacion_silhouette.png
│   ├── 13_heatmap_perfiles.png
│   └── 14_pipeline_completo.png
├── data/
│   └── ai4i_predictive_maintenance.csv
├── notebooks/
│   └── S3_ModelosNoSupervisados_AI4I.ipynb
└── README.md
```

---

## Entorno de Trabajo

| Herramienta | Versión |
|---|---|
| Python | 3.10+ |
| pandas | 2.x |
| numpy | 1.x |
| scikit-learn | 1.x |
| matplotlib | 3.x |
| seaborn | 0.13.x |
| Jupyter Notebook | 7.x |

### Instalación de dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## Metodología

El proyecto sigue un pipeline estructurado de aprendizaje no supervisado:

### 1. Análisis Exploratorio (EDA)

- Estadísticas descriptivas de las 5 variables de sensores
- Análisis de distribuciones mediante histogramas con medias anotadas
- Boxplots diferenciados por tipo de máquina (L, M, H)
- Matriz de correlación completa (variables + Machine failure)
- Pairplot entre variables clave coloreado por estado de fallo

**Hallazgo clave:** Correlación negativa fuerte entre Torque y Velocidad Rotacional (-0.88), y correlación positiva entre temperaturas del aire y del proceso (0.88).

### 2. Preprocesamiento

- Exclusión de la columna `UDI` (identificador sin valor predictivo)
- La variable `Machine failure` se conserva como referencia pero no se incluye en el clustering
- Codificación de la variable `Type` con `LabelEncoder`
- Escalado con `StandardScaler` para garantizar que ninguna variable domine el cálculo de distancias euclideas

### 3. K-Means

- Evaluación de K entre 2 y 10 mediante método del codo (inercia) y coeficiente de silueta
- K óptimo determinado en **K = 2** con Silhouette Score de **0.2109**
- Entrenamiento final sobre el dataset completo (10 000 registros)
- Visualización de clusters en 3 combinaciones de variables

### 4. DBSCAN

- Selección de `eps` mediante K-Distance Graph (distancia al 5° vecino más cercano)
- Parámetros finales: `eps = 0.8`, `min_samples = 5`
- Resultado: **2 clusters** y **261 puntos de ruido** (2.6% del total)
- Silhouette Score (excluyendo ruido): **0.3197**

### 5. PCA

- Reducción de 5 dimensiones a 2 componentes principales
- PC1 explica el **38.0%** de la varianza; PC2 el **20.4%** (acumulado: **58.4%**)
- Visualización de clusters K-Means en el espacio PCA
- Análisis de loadings para interpretación de componentes

### 6. t-SNE

- Reducción no lineal sobre muestra aleatoria de 2 000 puntos
- Parámetros: `perplexity=30`, `learning_rate=200`, `max_iter=1000`
- Visualización dual: clusters K-Means e intensidad de torque como gradiente de color
- Revela sub-agrupaciones internas no detectables con PCA

### 7. Detección de Anomalías

- **Isolation Forest** (`contamination=0.05`): 500 anomalías detectadas
- **Local Outlier Factor** (`n_neighbors=20`, `contamination=0.05`): 500 anomalías detectadas
- Análisis de consenso entre ambos métodos
- Validación cruzada con la variable `Machine failure`

---

## Resultados Principales

### Perfiles Operacionales Identificados (K-Means, K=2)

| Cluster | N° Máquinas | Temp. Aire [K] | Temp. Proceso [K] | Veloc. [rpm] | Torque [Nm] | Desgaste [min] | Tasa Fallo |
|---|---|---|---|---|---|---|---|
| C0 — Régimen Caliente | 4 955 | 301.6 | 311.8 | 1 535 | 39.8 | 126.8 | 3.3% |
| C1 — Régimen Estándar | 5 045 | 298.5 | 308.3 | 1 533 | 40.4 | 124.1 | 3.4% |

### Comparación de Algoritmos de Clustering

| Algoritmo | Parámetros | Clusters | Outliers | Silhouette | Forma |
|---|---|---|---|---|---|
| K-Means | k=2 | 2 | 0 | 0.2109 | Esférica |
| DBSCAN | eps=0.8, min_samples=5 | 2 | 261 | 0.3197 | Arbitraria |

### Visualizaciones Generadas

| Archivo | Descripción |
|---|---|
| `01_distribucion_variables.png` | Histogramas de las 5 variables de sensores y distribución por tipo |
| `02_boxplots_tipo_maquina.png` | Boxplots comparativos entre tipos L, M y H |
| `03_matriz_correlacion.png` | Heatmap de correlaciones entre variables numéricas |
| `04_pairplot.png` | Relaciones bivariadas coloreadas por estado de fallo |
| `05_seleccion_k.png` | Método del codo y silhouette score para selección de K |
| `06_kmeans_clusters.png` | Segmentación K-Means en tres proyecciones bidimensionales |
| `07_kdistance_graph.png` | K-Distance Graph para selección de eps en DBSCAN |
| `08_dbscan_clusters.png` | Segmentación DBSCAN con identificación de puntos de ruido |
| `09_pca_visualizacion.png` | Varianza explicada, acumulada y clusters en espacio PCA |
| `10_tsne_visualizacion.png` | Proyección t-SNE por clusters y por intensidad de torque |
| `11_deteccion_anomalias.png` | Comparación Isolation Forest vs LOF |
| `12_comparacion_silhouette.png` | Gráfico comparativo de silhouette score entre algoritmos |
| `13_heatmap_perfiles.png` | Heatmap de valores medios por cluster y variable |
| `14_pipeline_completo.png` | Diagrama del pipeline completo del proyecto |

---

## Análisis Crítico

### Limitaciones

- El silhouette score moderado (0.21–0.32) refleja la naturaleza continua de los datos de sensores industriales, donde las fronteras entre regímenes operacionales no son abruptas sino graduales. Esta característica es inherente a los sistemas físicos de manufactura y no indica un fallo del método.
- PCA retiene el 58.4% de la varianza con 2 componentes, lo que implica una pérdida de información relevante al visualizar en 2D. Para análisis más exhaustivo se recomienda trabajar con 3-4 componentes (varianza acumulada ~80%).
- La ausencia de etiquetas detalladas de tipo de fallo (solo disponible `Machine failure` binario) limita la validación semántica de los clusters con eventos de fallo específicos (overheat, tool wear failure, etc.).

### Propuestas de Mejora

- Integrar variables de contexto como turno de trabajo, antigüedad de la máquina o historial de mantenimiento para enriquecer los perfiles de clustering.
- Aplicar técnicas de clustering jerárquico (dendrogramas) para explorar la estructura a múltiples niveles de granularidad.
- Usar los clusters como features adicionales en modelos supervisados (Random Forest, XGBoost) para mejorar la predicción de fallos.
- Implementar monitoreo en tiempo real con actualización incremental de los clusters usando Mini-Batch K-Means.

---

## Recursos de Apoyo

- Scikit-learn — Clustering: https://scikit-learn.org/stable/modules/clustering.html
- AI4I Dataset (UCI): https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- t-SNE: van der Maaten & Hinton (2008). Visualizing Data using t-SNE. JMLR.
- DBSCAN: Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters.
