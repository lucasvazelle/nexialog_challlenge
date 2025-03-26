# requirements:
# streamlit, pandas, pyarrow, plotly, geopandas, shapely, numpy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from datetime import datetime, timedelta, time

st.set_page_config(layout="wide", page_title="Anomalie Network Dashboard")

# --- Custom CSS for monitoring tool feel ---
st.markdown("""
    <style>
    .main {
        background-color: #f3f6fc;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3, .st-bb, .st-at {
        color: #1f3b4d;
    }
    .metric-box {
        display: inline-block;
        margin: 0.5rem;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4BB Anomalie Network Dashboard")

# --- Chargement des données ---
@st.cache_data

def load_data():
    df = pd.read_parquet("../data/data_with_anomalies_with_jump_model.parquet")
    df["date_hour"] = pd.to_datetime(df["date_hour"])
    # Suppression des colonnes inutiles
    columns_to_drop = [
        "missing_avg_dns_time", "missing_std_dns_time",
        "missing_avg_latence_scoring", "missing_std_latence_scoring",
        "missing_avg_score_scoring", "missing_std_score_scoring",
        "encodage_heure_sin", "encodage_heure_cos",
        "encodage_jour_semaine_sin", "encodage_jour_semaine_cos"
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
    return df

df = load_data()

# --- Sidebar : Filtres temporels ---
st.sidebar.header("\u23F0 Filtrage temporel")
min_date = datetime(2024, 12, 3)
max_date = datetime(2025, 1, 31)
selected_date = st.sidebar.date_input("Date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
hour_options = [time(h, 0) for h in range(24)]
selected_time = st.sidebar.selectbox("Heure (pleine)", hour_options, index=datetime.now().hour)
selected_datetime = datetime.combine(selected_date, selected_time)
next_hour = selected_datetime + timedelta(hours=1)

# Choix période d'analyse
period_option = st.sidebar.selectbox("Analyser les anomalies depuis :", ["1 heure", "24 heures", "7 jours", "30 jours"])
if period_option == "1 heure":
    start_datetime = selected_datetime - timedelta(hours=1)
elif period_option == "24 heures":
    start_datetime = selected_datetime - timedelta(days=1)
elif period_option == "7 jours":
    start_datetime = selected_datetime - timedelta(days=7)
else:
    start_datetime = selected_datetime - timedelta(days=30)

# --- Filtrage des anomalies ---
st.sidebar.header("\u26A1 Type d'anomalie")
anomaly_type = st.sidebar.radio("Type d'anomalie", ["DNS", "Latence Scoring", "Score Scoring"])
anomaly_col_map = {
    "DNS": "is_jump_avg_dns_time",
    "Latence Scoring": "is_jump_avg_latence_scoring",
    "Score Scoring": "is_jump_avg_score_scoring"
}
selected_anomaly_col = anomaly_col_map[anomaly_type]

# --- Tabs ---
tabs = st.tabs(["\U0001F4CB Journal", "\U0001F5FA Carte", "\U0001F4CA Visualisation", "\U0001F6A8 Dashboard"])

# --- Journal ---
with tabs[0]:
    st.header("\U0001F4CB Journal des anomalies")
    journal_df = df[(df["date_hour"] >= selected_datetime) & (df["date_hour"] < next_hour)]
    journal_df = journal_df[journal_df[selected_anomaly_col]]
    st.dataframe(journal_df, use_container_width=True)

# --- Carte ---
with tabs[1]:
    st.header("\U0001F5FA Carte des anomalies")

    historical_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] <= selected_datetime)].copy()
    historical_df = historical_df.sort_values("date_hour")
    historical_df["anomalie"] = historical_df[selected_anomaly_col]

    gdf_departements = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    anomalies_by_dep = historical_df[historical_df["anomalie"]].groupby("code_departement").size().reset_index(name="nb_anomalies")
    anomalies_by_dep["code_departement"] = anomalies_by_dep["code_departement"].astype(str).str.zfill(2)
    gdf_departements["code"] = gdf_departements["code"]

    carte_df = gdf_departements.merge(anomalies_by_dep, how="left", left_on="code", right_on="code_departement")
    carte_df["nb_anomalies"] = carte_df["nb_anomalies"].fillna(0)
    carte_df["hover"] = carte_df["nom"] + " : " + carte_df["nb_anomalies"].astype(int).astype(str) + " anomalies"

    st.markdown(f"**Anomalies comptabilisées entre {start_datetime.strftime('%Y-%m-%d %H:%M')} et {selected_datetime.strftime('%Y-%m-%d %H:%M')}**")

    fig_map = px.choropleth_mapbox(
        carte_df,
        geojson=carte_df.geometry.__geo_interface__,
        locations=carte_df.index,
        color="nb_anomalies",
        hover_name="hover",
        mapbox_style="carto-positron",
        center={"lat": 46.6, "lon": 2.6},
        zoom=4.5,
        opacity=0.6,
        color_continuous_scale="YlOrRd",
        height=600,
    )

    st.plotly_chart(fig_map, use_container_width=True)

# --- Visualisation ---
with tabs[2]:
    st.header("\U0001F4CA Visualisation des OLTs")

    hour_df = df[(df["date_hour"] >= selected_datetime) & (df["date_hour"] < next_hour)]
    hour_df = hour_df[hour_df[selected_anomaly_col]]

    anomaly_olts = hour_df["olt_name"].unique().tolist()
    if anomaly_olts:
        selected_olt = st.selectbox("Sélectionner un OLT", sorted(anomaly_olts))
        olt_df = df[(df["olt_name"] == selected_olt) & (df["date_hour"] <= selected_datetime)].sort_values("date_hour")

        y_col = "avg_dns_time"
        if selected_anomaly_col == "is_jump_avg_latence_scoring":
            y_col = "avg_latence_scoring"
        elif selected_anomaly_col == "is_jump_avg_score_scoring":
            y_col = "avg_score_scoring"

        fig_ts = px.line(olt_df, x="date_hour", y=y_col, title=f"\U0001F4C8 {y_col} pour {selected_olt}", markers=True)
        anomalies_ts = olt_df[olt_df[selected_anomaly_col]]
        fig_ts.add_scatter(
            x=anomalies_ts["date_hour"],
            y=anomalies_ts[y_col],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomalies"
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Aucune anomalie détectée dans cette heure.")

# --- Alertes / Dashboard ---
with tabs[3]:
    st.header("\U0001F6A8 Statut des alertes")
    historical_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] <= selected_datetime)]
    historical_df["anomalie"] = historical_df[selected_anomaly_col]

    cols = st.columns(4)
    for i, attr in enumerate(["code_departement", "type_olt_model", "type_boucle", "boucle"]):
        counts = historical_df[historical_df["anomalie"]].groupby(attr).size().reset_index(name="nb_alertes")
        with cols[i]:
            st.markdown(f"### {attr}")
            for _, row in counts.iterrows():
                st.markdown(f"<div class='metric-box'><strong>{row[attr]}</strong><br/>{row['nb_alertes']} alertes</div>", unsafe_allow_html=True)

st.caption("Dernière mise à jour des données : {}".format(max_date.strftime("%Y-%m-%d %H:%M:%S")))