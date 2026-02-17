"""
Customer Lifetime Value & Risk Dashboard
TFM - Master en Data Science
Autor: Cesar Gonzalez Franco

Ejecutar:
    streamlit run app_clv_dashboard.py

Compatible con Streamlit Community Cloud.
"""

import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from lifetimes import BetaGeoFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

# Ruta base del proyecto (donde esta este archivo)
BASE_DIR = pathlib.Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "_csv"
MODELS_DIR = BASE_DIR / "_models"

# ─── CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV & Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rosé Pine Moon colors
COLORS = {
    "bg":       "#232136",
    "surface":  "#2a273f",
    "text":     "#e0def4",
    "muted":    "#908caa",
    "rose":     "#eb6f92",
    "pine":     "#3e8fb0",
    "foam":     "#9ccfd8",
    "iris":     "#c4a7e7",
    "gold":     "#f6c177",
    "love":     "#ea9a97",
}

SEGMENT_COLORS = {
    "Premium":    COLORS["gold"],
    "Potential":  COLORS["iris"],
    "Occasional": COLORS["pine"],
    "Lost":       COLORS["rose"],
}

# ─── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_DIR / "CLV_Churn_by_Customer.csv")
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df

df = load_data()

# ─── LOAD MODELS ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Carga los modelos serializados desde _models/ (formato joblib)."""
    # BG/NBD — reconstruir desde atributos (evita dill)
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf_attrs = joblib.load(MODELS_DIR / "bgf_model.pkl")
    bgf.__dict__.update(bgf_attrs)
    bgf.predict = bgf.conditional_expected_number_of_purchases_up_to_time

    # Gamma-Gamma — reconstruir desde atributos
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf_attrs = joblib.load(MODELS_DIR / "ggf_model.pkl")
    ggf.__dict__.update(ggf_attrs)

    km = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    sc = joblib.load(MODELS_DIR / "scaler.pkl")
    return bgf, ggf, km, sc

models_loaded = False
try:
    bgf_model, ggf_model, kmeans_model, scaler_model = load_models()
    models_loaded = True
except Exception:
    pass  # Tab 5 mostrara aviso si no hay modelos

# ─── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Filtros")

# Segment filter
segments = ["Todos"] + sorted(df["segment"].dropna().unique().tolist())
selected_segment = st.sidebar.selectbox("Segmento", segments)

# Status filter
status_options = ["Todos", "Activo", "En riesgo"]
selected_status = st.sidebar.selectbox("Estado de actividad", status_options)

# P(alive) range
p_alive_range = st.sidebar.slider(
    "Rango de P(alive)",
    min_value=0.0, max_value=1.0,
    value=(0.0, 1.0), step=0.01
)

# Apply filters
df_filtered = df.copy()
if selected_segment != "Todos":
    df_filtered = df_filtered[df_filtered["segment"] == selected_segment]
if selected_status == "Activo":
    df_filtered = df_filtered[df_filtered["churn"] == 0]
elif selected_status == "En riesgo":
    df_filtered = df_filtered[df_filtered["churn"] == 1]
df_filtered = df_filtered[
    (df_filtered["p_alive"] >= p_alive_range[0]) &
    (df_filtered["p_alive"] <= p_alive_range[1])
]

# ─── HEADER ───────────────────────────────────────────────────
st.title("📊 Customer Lifetime Value & Risk Dashboard")
st.caption("TFM — Modelizacion del Ciclo de Vida del Cliente | BG/NBD + Gamma-Gamma")

# ─── KPI ROW ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_clients = len(df_filtered)
total_at_risk = df_filtered["churn"].sum()
pct_risk = (total_at_risk / total_clients * 100) if total_clients > 0 else 0
total_clv = df_filtered["clv"].sum()
clv_at_risk = df_filtered[df_filtered["churn"] == 1]["clv"].sum()
avg_p_alive = df_filtered["p_alive"].mean()

col1.metric("Total Clientes", f"{total_clients:,}")
col2.metric("En Riesgo", f"{total_at_risk:,}", f"{pct_risk:.1f}%")
col3.metric("CLV Total", f"£{total_clv:,.0f}")
col4.metric("CLV en Riesgo", f"£{clv_at_risk:,.0f}", f"-{clv_at_risk/total_clv*100:.1f}%" if total_clv > 0 else "0%")
col5.metric("P(alive) Media", f"{avg_p_alive:.3f}")

st.divider()

# ─── TAB LAYOUT ───────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Vision General",
    "👥 Tabla de Clientes",
    "🔍 Ficha de Cliente",
    "💡 Simulador de Retencion",
    "🔮 Scoring Nuevo Cliente"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: VISION GENERAL
# ═══════════════════════════════════════════════════════════════
with tab1:
    row1_col1, row1_col2 = st.columns(2)

    # P(alive) distribution
    with row1_col1:
        fig_hist = px.histogram(
            df_filtered, x="p_alive", nbins=50,
            color="churn_label",
            color_discrete_map={"Activo": COLORS["pine"], "Churned": COLORS["rose"]},
            title="Distribucion de P(alive)",
            labels={"p_alive": "P(alive)", "count": "Clientes", "churn_label": "Estado"},
            opacity=0.75
        )
        fig_hist.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font_color=COLORS["text"], bargap=0.05,
            legend=dict(title="", orientation="h", y=1.12)
        )
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color=COLORS["gold"],
                          annotation_text="Umbral: 0.7")
        st.plotly_chart(fig_hist, use_container_width=True)

    # CLV by segment
    with row1_col2:
        seg_summary = df_filtered.groupby("segment").agg(
            CLV_medio=("clv", "mean"),
            CLV_total=("clv", "sum"),
            N_clientes=("clv", "count"),
            P_alive_media=("p_alive", "mean")
        ).reset_index()

        fig_bar = px.bar(
            seg_summary.sort_values("CLV_medio", ascending=True),
            x="CLV_medio", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="CLV Medio por Segmento",
            labels={"CLV_medio": "CLV Medio (£)", "segment": ""},
            text=seg_summary.sort_values("CLV_medio", ascending=True)["CLV_medio"].apply(lambda x: f"£{x:,.0f}")
        )
        fig_bar.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font_color=COLORS["text"], showlegend=False
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    # Risk rate by segment
    with row2_col1:
        risk_by_seg = df_filtered.groupby("segment").agg(
            N_total=("churn", "count"),
            N_riesgo=("churn", "sum")
        ).reset_index()
        risk_by_seg["Tasa_riesgo"] = (risk_by_seg["N_riesgo"] / risk_by_seg["N_total"] * 100).round(1)

        fig_risk = px.bar(
            risk_by_seg.sort_values("Tasa_riesgo", ascending=True),
            x="Tasa_riesgo", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Riesgo de Inactividad por Segmento (BG/NBD)",
            labels={"Tasa_riesgo": "Clientes en riesgo (%)", "segment": ""},
            text=risk_by_seg.sort_values("Tasa_riesgo", ascending=True)["Tasa_riesgo"].apply(lambda x: f"{x:.1f}%")
        )
        fig_risk.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font_color=COLORS["text"], showlegend=False
        )
        fig_risk.update_traces(textposition="outside")
        st.plotly_chart(fig_risk, use_container_width=True)

    # CLV at risk by segment
    with row2_col2:
        clv_risk_seg = df_filtered[df_filtered["churn"] == 1].groupby("segment").agg(
            CLV_riesgo=("clv", "sum")
        ).reset_index()

        if not clv_risk_seg.empty:
            fig_clv_risk = px.bar(
                clv_risk_seg.sort_values("CLV_riesgo", ascending=True),
                x="CLV_riesgo", y="segment", orientation="h",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                title="CLV Total en Riesgo por Segmento",
                labels={"CLV_riesgo": "CLV en Riesgo (£)", "segment": ""},
                text=clv_risk_seg.sort_values("CLV_riesgo", ascending=True)["CLV_riesgo"].apply(lambda x: f"£{x:,.0f}")
            )
            fig_clv_risk.update_layout(
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font_color=COLORS["text"], showlegend=False
            )
            fig_clv_risk.update_traces(textposition="outside")
            st.plotly_chart(fig_clv_risk, use_container_width=True)

    # Scatter: P(alive) vs CLV
    st.subheader("P(alive) vs CLV por Segmento")
    fig_scatter = px.scatter(
        df_filtered[df_filtered["clv"] > 0],
        x="p_alive", y="clv", color="segment",
        color_discrete_map=SEGMENT_COLORS,
        size="predicted_purchases", size_max=15,
        opacity=0.5, log_y=True,
        labels={"p_alive": "P(alive)", "clv": "CLV (£, log)", "segment": "Segmento",
                "predicted_purchases": "Compras predichas"},
        title=""
    )
    fig_scatter.add_vline(x=0.7, line_dash="dash", line_color=COLORS["gold"],
                          annotation_text="Umbral riesgo")
    fig_scatter.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
        font_color=COLORS["text"], height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: TABLA DE CLIENTES
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Explorador de Clientes")

    # Sort options
    sort_col = st.selectbox("Ordenar por:", ["clv", "p_alive", "frequency_cal", "recency_cal"], index=0)
    sort_asc = st.checkbox("Orden ascendente", value=False)

    display_df = df_filtered[[
        "Customer ID", "segment", "p_alive", "churn_label",
        "frequency_cal", "recency_cal", "monetary_value_cal",
        "clv", "predicted_purchases"
    ]].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    display_df.columns = [
        "ID Cliente", "Segmento", "P(alive)", "Estado",
        "Frecuencia", "Recencia (dias)", "Valor Monetario (£)",
        "CLV (£)", "Compras Predichas"
    ]

    # Format
    display_df["P(alive)"] = display_df["P(alive)"].round(4)
    display_df["CLV (£)"] = display_df["CLV (£)"].round(2)
    display_df["Compras Predichas"] = display_df["Compras Predichas"].round(2)
    display_df["Valor Monetario (£)"] = display_df["Valor Monetario (£)"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config={
            "P(alive)": st.column_config.ProgressColumn(
                "P(alive)", min_value=0, max_value=1, format="%.3f"
            ),
            "CLV (£)": st.column_config.NumberColumn("CLV (£)", format="£%.2f"),
            "Valor Monetario (£)": st.column_config.NumberColumn("Valor Mon. (£)", format="£%.2f"),
        }
    )

    st.caption(f"Mostrando {len(display_df):,} clientes")


# ═══════════════════════════════════════════════════════════════
# TAB 3: FICHA DE CLIENTE
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ficha Individual de Cliente")

    customer_id = st.number_input(
        "Introduce el ID del cliente:",
        min_value=int(df["Customer ID"].min()),
        max_value=int(df["Customer ID"].max()),
        value=int(df.nlargest(1, "clv")["Customer ID"].values[0]),
        step=1
    )

    client = df[df["Customer ID"] == customer_id]

    if client.empty:
        st.warning(f"Cliente {customer_id} no encontrado en la base de datos.")
    else:
        c = client.iloc[0]

        # Status indicator
        if c["churn"] == 0:
            st.success(f"🟢 Cliente ACTIVO — P(alive) = {c['p_alive']:.4f}")
        else:
            st.error(f"🔴 Cliente EN RIESGO — P(alive) = {c['p_alive']:.4f}")

        # Client metrics
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Segmento", c["segment"])
        fc2.metric("CLV (12 meses)", f"£{c['clv']:,.2f}")
        fc3.metric("Compras Predichas", f"{c['predicted_purchases']:.1f}")
        fc4.metric("P(alive)", f"{c['p_alive']:.4f}")

        fc5, fc6, fc7, fc8 = st.columns(4)
        fc5.metric("Frecuencia", f"{c['frequency_cal']:.0f} compras")
        fc6.metric("Recencia", f"{c['recency_cal']:.0f} dias")
        fc7.metric("Valor Monetario Medio", f"£{c['monetary_value_cal']:,.2f}")
        fc8.metric("Antiguedad (T)", f"{c['T_cal']:.0f} dias")

        # Gauge chart for P(alive)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=c["p_alive"],
            number={"suffix": "", "font": {"size": 40, "color": COLORS["text"]}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": COLORS["muted"]},
                "bar": {"color": COLORS["pine"] if c["churn"] == 0 else COLORS["rose"]},
                "bgcolor": COLORS["surface"],
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(235,111,146,0.3)"},
                    {"range": [0.3, 0.7], "color": "rgba(246,193,119,0.3)"},
                    {"range": [0.7, 1.0], "color": "rgba(62,143,176,0.3)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["gold"], "width": 3},
                    "thickness": 0.8,
                    "value": 0.7
                }
            },
            title={"text": "Probabilidad de Actividad", "font": {"color": COLORS["text"]}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor=COLORS["bg"],
            font_color=COLORS["text"],
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Context: where does this client stand?
        st.caption(
            f"Este cliente se encuentra en el percentil "
            f"**{(df['clv'] <= c['clv']).mean()*100:.0f}** de CLV "
            f"y en el percentil **{(df['p_alive'] <= c['p_alive']).mean()*100:.0f}** de P(alive) "
            f"respecto a la base total de {len(df):,} clientes."
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4: SIMULADOR DE RETENCION
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Simulador de Impacto de Retencion")
    st.markdown(
        "Este simulador estima el **CLV recuperable** si se aplican campanas "
        "de retencion con una tasa de exito determinada sobre los clientes en riesgo."
    )

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        target_segments = st.multiselect(
            "Segmentos objetivo:",
            options=sorted(df["segment"].dropna().unique()),
            default=sorted(df["segment"].dropna().unique())
        )

    with sim_col2:
        retention_rate = st.slider(
            "Tasa de exito de la campana (%)",
            min_value=5, max_value=100, value=30, step=5
        )

    # Calculate
    at_risk = df[(df["churn"] == 1) & (df["segment"].isin(target_segments))]

    if at_risk.empty:
        st.info("No hay clientes en riesgo en los segmentos seleccionados.")
    else:
        results = []
        for seg in target_segments:
            seg_risk = at_risk[at_risk["segment"] == seg]
            n_risk = len(seg_risk)
            clv_total_risk = seg_risk["clv"].sum()
            n_retained = int(np.ceil(n_risk * retention_rate / 100))
            # Assume we retain the highest-CLV clients first
            clv_recovered = seg_risk.nlargest(n_retained, "clv")["clv"].sum()

            results.append({
                "Segmento": seg,
                "Clientes en riesgo": n_risk,
                "CLV total en riesgo (£)": clv_total_risk,
                "Clientes retenidos": n_retained,
                "CLV recuperado (£)": clv_recovered,
                "ROI potencial (£)": clv_recovered
            })

        results_df = pd.DataFrame(results)

        # Summary metrics
        total_recovered = results_df["CLV recuperado (£)"].sum()
        total_risk_clv = results_df["CLV total en riesgo (£)"].sum()
        total_retained = results_df["Clientes retenidos"].sum()

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Clientes retenidos", f"{total_retained:,}")
        sm2.metric("CLV recuperado", f"£{total_recovered:,.0f}")
        sm3.metric("% del CLV en riesgo salvado", f"{total_recovered/total_risk_clv*100:.1f}%" if total_risk_clv > 0 else "0%")

        # Results table
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "CLV total en riesgo (£)": st.column_config.NumberColumn(format="£%.0f"),
                "CLV recuperado (£)": st.column_config.NumberColumn(format="£%.0f"),
                "ROI potencial (£)": st.column_config.NumberColumn(format="£%.0f"),
            },
            hide_index=True
        )

        # Visualization
        fig_sim = px.bar(
            results_df,
            x="Segmento", y=["CLV total en riesgo (£)", "CLV recuperado (£)"],
            barmode="group",
            color_discrete_sequence=[COLORS["rose"], COLORS["pine"]],
            title=f"CLV en Riesgo vs CLV Recuperable (tasa de exito: {retention_rate}%)",
            labels={"value": "CLV (£)", "variable": ""}
        )
        fig_sim.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font_color=COLORS["text"],
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown(
            f"**Interpretacion:** Con una tasa de exito del **{retention_rate}%** "
            f"sobre los clientes en riesgo de los segmentos seleccionados, "
            f"se podrian retener **{total_retained:,} clientes** y recuperar "
            f"**£{total_recovered:,.0f}** de CLV, equivalente al "
            f"**{total_recovered/total_risk_clv*100:.1f}%** del valor total en riesgo."
        )


# ═══════════════════════════════════════════════════════════════
# TAB 5: SCORING DE NUEVO CLIENTE
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Scoring en Tiempo Real con Modelos Entrenados")
    st.markdown(
        "Introduce los datos transaccionales de un cliente y el sistema calculara "
        "su **P(alive)**, **compras esperadas**, **CLV** y **segmento** usando los "
        "modelos BG/NBD, Gamma-Gamma y K-Means entrenados."
    )

    if not models_loaded:
        st.error(
            "⚠️ No se encontraron los modelos entrenados en `_models/`. "
            "Ejecuta el notebook `_tfm_v6.ipynb` completo para generar los archivos "
            "`.pkl` necesarios."
        )
    else:
        # ─── Input form ───
        st.markdown("#### Datos del cliente")
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            inp_frequency = st.number_input(
                "Frequency (compras repetidas)",
                min_value=0, max_value=500, value=5, step=1,
                help="Numero de compras repetidas (excluyendo la primera)"
            )
            inp_recency = st.number_input(
                "Recency (dias desde 1a a ultima compra)",
                min_value=0, max_value=1000, value=200, step=1,
                help="Dias entre la primera y la ultima compra del cliente"
            )

        with input_col2:
            inp_T = st.number_input(
                "T (antiguedad en dias)",
                min_value=1, max_value=1000, value=365, step=1,
                help="Dias desde la primera compra hasta la fecha de corte"
            )
            inp_monetary = st.number_input(
                "Monetary Value (valor medio por transaccion, £)",
                min_value=0.01, max_value=100000.0, value=250.0, step=10.0,
                help="Valor monetario promedio por transaccion"
            )

        CHURN_THRESHOLD = 0.7

        if st.button("🚀 Calcular", type="primary", use_container_width=True):

            # ─── P(alive) ───
            p_alive = float(np.squeeze(bgf_model.conditional_probability_alive(
                inp_frequency, inp_recency, inp_T
            )))

            # ─── Compras esperadas (12 meses = 365 dias) ───
            if inp_frequency > 0:
                expected_purchases = float(np.squeeze(bgf_model.conditional_expected_number_of_purchases_up_to_time(
                    365, inp_frequency, inp_recency, inp_T
                )))
            else:
                expected_purchases = 0.0

            # ─── CLV (12 meses, discount_rate=1% mensual) ───
            if inp_frequency > 0 and inp_monetary > 0:
                clv = ggf_model.customer_lifetime_value(
                    bgf_model,
                    frequency=pd.Series([inp_frequency]),
                    recency=pd.Series([inp_recency]),
                    T=pd.Series([inp_T]),
                    monetary_value=pd.Series([inp_monetary]),
                    time=12,
                    discount_rate=0.01,
                    freq="D"
                ).values[0]
            else:
                clv = 0.0

            # ─── Segmento via KMeans ───
            log_vals = np.log1p([inp_recency, inp_frequency, inp_monetary]).reshape(1, -1)
            scaled_vals = scaler_model.transform(log_vals)
            cluster_id = kmeans_model.predict(scaled_vals)[0]

            segment_map = {0: "Occasional", 1: "Lost", 2: "Premium", 3: "Potential"}
            segment = segment_map.get(cluster_id, f"Cluster {cluster_id}")

            is_at_risk = p_alive < CHURN_THRESHOLD

            # ─── Results ───
            st.divider()
            st.markdown("#### Resultados del Scoring")

            if is_at_risk:
                st.error(f"🔴 **EN RIESGO** — P(alive) = {p_alive:.4f} (< {CHURN_THRESHOLD})")
            else:
                st.success(f"🟢 **ACTIVO** — P(alive) = {p_alive:.4f} (≥ {CHURN_THRESHOLD})")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("P(alive)", f"{p_alive:.4f}")
            res_col2.metric("Compras esperadas (12m)", f"{expected_purchases:.1f}")
            res_col3.metric("CLV (12 meses)", f"£{clv:,.2f}")
            res_col4.metric("Segmento", segment)

            # ─── Gauge chart ───
            fig_gauge_score = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_alive,
                number={"suffix": "", "font": {"size": 40, "color": COLORS["text"]}},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": COLORS["muted"]},
                    "bar": {"color": COLORS["pine"] if not is_at_risk else COLORS["rose"]},
                    "bgcolor": COLORS["surface"],
                    "steps": [
                        {"range": [0, 0.3], "color": "rgba(235,111,146,0.3)"},
                        {"range": [0.3, 0.7], "color": "rgba(246,193,119,0.3)"},
                        {"range": [0.7, 1.0], "color": "rgba(62,143,176,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["gold"], "width": 3},
                        "thickness": 0.8,
                        "value": CHURN_THRESHOLD
                    }
                },
                title={"text": "Probabilidad de Actividad", "font": {"color": COLORS["text"]}}
            ))
            fig_gauge_score.update_layout(
                paper_bgcolor=COLORS["bg"],
                font_color=COLORS["text"],
                height=300
            )
            st.plotly_chart(fig_gauge_score, use_container_width=True)

            # ─── Context vs base ───
            pct_clv = (df["clv"] <= clv).mean() * 100
            pct_palive = (df["p_alive"] <= p_alive).mean() * 100
            st.caption(
                f"Este perfil se situa en el percentil **{pct_clv:.0f}** de CLV "
                f"y en el percentil **{pct_palive:.0f}** de P(alive) "
                f"respecto a la base de {len(df):,} clientes."
            )

            # ─── Modelos utilizados ───
            with st.expander("ℹ️ Modelos utilizados"):
                st.markdown(
                    f"- **BG/NBD** (`bgf_model.pkl`): r={bgf_model.params_['r']:.4f}, "
                    f"α={bgf_model.params_['alpha']:.4f}, "
                    f"a={bgf_model.params_['a']:.4f}, b={bgf_model.params_['b']:.4f}\n"
                    f"- **Gamma-Gamma** (`ggf_model.pkl`): p={ggf_model.params_['p']:.4f}, "
                    f"q={ggf_model.params_['q']:.4f}, v={ggf_model.params_['v']:.4f}\n"
                    f"- **K-Means** (`kmeans_model.pkl`): K=4 clusters\n"
                    f"- **Scaler** (`scaler.pkl`): StandardScaler sobre log(RFM)\n"
                    f"- **Umbral de riesgo**: P(alive) < {CHURN_THRESHOLD}"
                )


# ─── FOOTER ───────────────────────────────────────────────────
st.divider()
st.caption(
    "Dashboard desarrollado como parte del TFM: "
    "'Modelizacion del Ciclo de Vida del Cliente en E-commerce' | "
    "Modelos: BG/NBD + Gamma-Gamma | "
    "Datos: Online Retail II (UCI ML Repository)"
)
