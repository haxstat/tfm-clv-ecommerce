# Customer Lifetime Value & Risk Analysis in E-Commerce

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tfm-clv-ecommerce.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green)

> **Trabajo Fin de Máster (TFM)** — Máster en Data Science
> **Autor:** César González Franco



---

## Descripción

Proyecto de investigación académica que implementa un pipeline completo de **Customer Lifetime Value (CLV)** y **análisis de riesgo de inactividad** sobre datos reales de e-commerce (Online Retail II, UCI Machine Learning Repository).

El estudio combina segmentación no supervisada (KMeans), modelos probabilísticos del comportamiento de compra (BG/NBD, Gamma-Gamma) y un dashboard interactivo desplegado en la nube para la toma de decisiones comerciales.

### Demo en vivo

**[Abrir Dashboard](https://tfm-clv-ecommerce.streamlit.app/)**

---

## Estructura del Repositorio

```
tfm-clv-ecommerce/
|
|-- app_clv_dashboard.py          # Dashboard interactivo (Streamlit)
|-- requirements.txt              # Dependencias Python
|-- _TFM_cesargonzalezfranco.ipynb # Notebook con el análisis completo
|
|-- _csv/                         # Datasets generados
|   |-- CLV_Churn_by_Customer.csv    # Dataset principal (4,822 clientes)
|   |-- CLV_by_Customer.csv          # CLV individual
|   |-- CLV_by_Cluster_Summary.csv   # Resumen CLV por segmento
|   |-- Top_100_CLV.csv              # Top 100 clientes por valor
|   +-- Churn_by_Segment_Summary.csv # Riesgo por segmento
|
|-- _models/                      # Modelos serializados (joblib)
|   |-- bgf_model.pkl               # BG/NBD (BetaGeoFitter)
|   |-- ggf_model.pkl               # Gamma-Gamma (GammaGammaFitter)
|   |-- kmeans_model.pkl             # KMeans (K=4)
|   +-- scaler.pkl                   # StandardScaler
|
+-- _img/                         # Visualizaciones generadas (29 PNGs)
    |-- EDA_*.png                    # Análisis exploratorio
    |-- RFM_*.png                    # Distribuciones RFM
    |-- KMeans_*.png                 # Métricas de clustering
    |-- CLV_*.png                    # Análisis de valor de vida
    |-- Churn_*.png                  # Análisis de riesgo
    +-- P_alive_*.png                # Probabilidad de actividad
```

---

## Metodología

El análisis sigue un pipeline secuencial de 8 fases:

### 1. Análisis Exploratorio (EDA)
- Dataset: **Online Retail II** (UCI ML Repository) — transacciones reales de un retailer UK (2009-2011)
- Limpieza de datos: tratamiento de valores nulos, devoluciones, cantidades/precios negativos
- Análisis temporal de facturación y ventas mensuales

### 2. Ingeniería de Características (RFM)
- Construcción de variables **Recency**, **Frequency** y **Monetary Value** por cliente
- Transformación logarítmica (`log1p`) para normalizar distribuciones sesgadas
- Estandarización con `StandardScaler` para clustering

### 3. Segmentación de Clientes (KMeans)
- Evaluación sistemática de K=2..10 con 4 métricas: Silhouette, Calinski-Harabasz, Davies-Bouldin e Inercia (Elbow)
- **K=4 óptimo** con segmentos interpretables:
  - **Premium**: alta frecuencia, alto valor, compra reciente
  - **Potential**: actividad moderada con potencial de crecimiento
  - **Occasional**: compras esporádicas y de bajo valor
  - **Lost**: inactivos con última compra lejana

### 4. Modelización Probabilística — BG/NBD
- Modelo **Beta-Geometric/Negative Binomial Distribution** para predecir comportamiento de compra futuro
- Parámetros ajustados: `r=0.641, alpha=62.873, a=0.024, b=0.322`
- Penalizador: `penalizer_coef=0.03`
- Predicción de: frecuencia de compra esperada y **P(alive)** por cliente

### 5. Modelización Probabilística — Gamma-Gamma
- Modelo **Gamma-Gamma** para estimar valor monetario esperado por transacción
- Parámetros ajustados: `p=2.323, q=3.468, v=434.688`
- Validación previa: correlación Frequency-Monetary < 0.3 (asunción de independencia)

### 6. Estimación del CLV
- **Customer Lifetime Value** a 12 meses con tasa de descuento mensual del 1%
- Integración de BG/NBD (frecuencia esperada) + Gamma-Gamma (valor esperado) + P(alive)
- Análisis de Pareto: concentración del valor en pocos clientes

### 7. Análisis de Riesgo de Inactividad
- Umbral de riesgo: **P(alive) < 0.70**
- Análisis de sensibilidad del umbral
- Distribución del riesgo por segmento
- Identificación de clientes en riesgo para acciones de retención

---

## Resultados Clave

| Métrica | Valor |
|---|---|
| Clientes analizados | **4,822** |
| Segmentos identificados | **4** (Premium, Potential, Occasional, Lost) |
| Umbral de riesgo P(alive) | **0.70** |
| Clientes en riesgo | **282** (5.8%) |
| CLV total estimado (12 meses) | **6,485,442 GBP** |

---

## Dashboard Interactivo

El dashboard consta de **5 pestañas**:

| Pestaña | Descripción |
|---|---|
| **Visión General** | KPIs globales, distribución de segmentos, CLV por cluster, métricas agregadas |
| **Tabla de Clientes** | Tabla filtrable con todos los clientes, métricas RFM, CLV, P(alive), segmento |
| **Ficha de Cliente** | Perfil individual detallado: gauge P(alive), indicadores de riesgo, contexto vs base |
| **Simulador de Retención** | Simulación de impacto económico de retener clientes en riesgo (escenarios what-if) |
| **Scoring Nuevo Cliente** | Scoring en tiempo real usando los modelos entrenados: ingresa RFM y obtiene P(alive), CLV, segmento |

**Tema visual:** Rose Pine Moon (dark theme)

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Análisis y modelización | Python, pandas, numpy, scikit-learn |
| Modelos probabilísticos | [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) (BG/NBD, Gamma-Gamma) |
| Visualización (notebook) | matplotlib, seaborn |
| Dashboard | Streamlit, Plotly |
| Serialización de modelos | joblib |
| Despliegue | Streamlit Community Cloud |
| Control de versiones | Git, GitHub |

---

## Ejecución Local

### Requisitos previos
- Python 3.10+
- pip o conda

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/haxstat/tfm-clv-ecommerce.git
cd tfm-clv-ecommerce

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run app_clv_dashboard.py
```

El dashboard estará disponible en `http://localhost:8501`

### Notebook
Para ejecutar el notebook de análisis completo, se necesita adicionalmente el dataset original `online_retail_II.xlsx` (no incluido en el repositorio por su tamaño). Disponible en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

---

## Autor

**César González Franco**
Máster en Data Science
GitHub: [@haxstat](https://github.com/haxstat)


## License

This project is published for academic purposes.

Dataset: Online Retail II – UCI Machine Learning Repository.
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).