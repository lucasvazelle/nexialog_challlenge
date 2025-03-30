import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import dash.dash_table
import io
import base64

# 2. Chargement des données
def load_data():
    df = pd.read_parquet("../data/data_with_anomalies_with_jump_model.parquet")
    df["date_hour"] = pd.to_datetime(df["date_hour"])
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

# 3. Options pour les filtres
boucle_options = [{'label': str(val), 'value': str(val)}
                  for val in sorted(df["boucle"].dropna().unique())]
olt_type_options = [{'label': str(val), 'value': str(val)}
                    for val in sorted(df["type_olt_model"].dropna().unique())]

# 4. Dates limites
min_date = datetime(2024, 12, 3)
max_date = datetime(2025, 1, 31)

# 5. Mapping pour les types d'anomalies
anomaly_col_map = {
    "DNS": "is_jump_avg_dns_time",
    "Latence Scoring": "is_jump_avg_latence_scoring",
    "Score Scoring": "is_jump_avg_score_scoring"
}

# 6. Fonctions de création de contenu pour les pages
def create_dashboard_content(historical_df, selected_anomaly_col):
    historical_df["anomalie"] = historical_df[selected_anomaly_col]
    attrs = ["code_departement", "type_olt_model", "type_boucle", "boucle"]
    
    # Statistiques globales
    total_anomalies = historical_df[historical_df["anomalie"]].shape[0]
    total_olts = historical_df[historical_df["anomalie"]]["olt_name"].nunique()
    
    stats_cards = html.Div([
        html.Div([
            html.H2(f"{total_anomalies}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Anomalies détectées", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "200px"}),
        html.Div([
            html.H2(f"{total_olts}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("OLTs affectés", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "200px"})
    ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"})
    
    # Créer des graphiques améliorés
    pie_charts = []
    for attr in attrs:
        counts = historical_df[historical_df["anomalie"]].groupby(attr).size().reset_index(name="nb_alertes")
        if attr in ["code_departement", "boucle"]:
            counts = counts.sort_values("nb_alertes", ascending=False).head(5)
        else:
            counts = counts.sort_values("nb_alertes", ascending=False).head(5)
        
        pie_fig = px.pie(
            counts,
            names=attr,
            values="nb_alertes",
            hole=0.5,
            title=f"Répartition par {attr.replace('_', ' ').title()}",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        pie_fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color='#E2E8F0', size=11)
            ),
            title=dict(
                font=dict(size=16, color='#60A5FA'),
                x=0.5
            ),
            paper_bgcolor='#1E293B',
            plot_bgcolor='#1E293B',
            font=dict(color='#E2E8F0')
        )
        
        pie_charts.append(html.Div(
            dcc.Graph(figure=pie_fig, config={'displayModeBar': False}),
            className="card",
            style={'width': 'calc(50% - 30px)', 'margin': '10px'}
        ))
    
    # Tendance temporelle des anomalies
    time_trend = historical_df[historical_df["anomalie"]].groupby(
        pd.Grouper(key='date_hour', freq='D')
    ).size().reset_index(name='count')
    
    trend_fig = px.line(
        time_trend, 
        x='date_hour', 
        y='count',
        template="plotly_dark",
        title="Évolution des anomalies dans le temps",
        labels={"count": "Nombre d'anomalies", "date_hour": "Date"}
    )
    
    trend_fig.update_traces(
        line=dict(color='#38BDF8', width=3),
        mode='lines+markers',
        marker=dict(size=8, color='#0EA5E9')
    )
    
    trend_fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=30),
        title=dict(
            font=dict(size=16, color='#60A5FA'),
            x=0.5
        ),
        paper_bgcolor='#1E293B',
        plot_bgcolor='#1E293B',
        font=dict(color='#E2E8F0'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(150, 150, 150, 0.1)',
            title_font=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(150, 150, 150, 0.1)',
            title_font=dict(size=12)
        ),
        height=300
    )
    
    trend_chart = html.Div(
        dcc.Graph(figure=trend_fig, config={'displayModeBar': False}),
        className="card",
        style={'width': 'calc(100% - 20px)', 'margin': '10px'}
    )
    
    return html.Div([
        html.H2("Tableau de bord des anomalies", className="page-title", style={"marginTop": "0"}),
        stats_cards,
        trend_chart,
        html.H3("Répartition des anomalies", className="section-title"),
        html.Div(pie_charts, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"})
    ])


def create_visual_content(olt_df, selected_olt, selected_anomaly_col):
    if not selected_olt:
        return html.Div([
            html.H2("Visualisation OLT", className="page-title", style={"marginTop": "0"}),
            html.Div([
                html.P("Veuillez sélectionner un OLT pour afficher les données", style={"color": "#CBD5E1"}),
                html.Div(html.I(className="fas fa-chart-line", style={"fontSize": "80px", "color": "#334155", "opacity": "0.7"}), 
                         style={"textAlign": "center", "margin": "40px 0"})
            ], style={"textAlign": "center", "padding": "40px"}, className="card")
        ])
    
    y_col = "avg_dns_time"
    metric_name = "DNS"
    if selected_anomaly_col == "is_jump_avg_latence_scoring":
        y_col = "avg_latence_scoring"
        metric_name = "Latence Scoring"
    elif selected_anomaly_col == "is_jump_avg_score_scoring":
        y_col = "avg_score_scoring"
        metric_name = "Score Scoring"
    
    # Statistiques pour l'OLT sélectionné
    anomalies_count = olt_df[olt_df[selected_anomaly_col]].shape[0]
    current_value = olt_df.iloc[-1][y_col] if not olt_df.empty else 0
    avg_value = olt_df[y_col].mean() if not olt_df.empty else 0
    max_value = olt_df[y_col].max() if not olt_df.empty else 0
    
    # Calcul de la variation en pourcentage
    prev_value = olt_df.iloc[-2][y_col] if len(olt_df) > 1 else current_value
    percent_change = ((current_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
    
    stats_cards = html.Div([
        html.Div([
            html.H3("OLT sélectionné", style={"margin": "0", "color": "#94A3B8", "fontSize": "1rem"}),
            html.H2(f"{selected_olt}", style={"margin": "5px 0", "color": "#E2E8F0", "fontSize": "1.5rem"})
        ], className="card", style={"width": "250px"}),
        html.Div([
            html.H3("Anomalies détectées", style={"margin": "0", "color": "#94A3B8", "fontSize": "1rem"}),
            html.H2(f"{anomalies_count}", style={"margin": "5px 0", "color": "#38BDF8", "fontSize": "1.5rem"})
        ], className="card", style={"width": "200px"}),
        html.Div([
            html.H3(f"Valeur actuelle {metric_name}", style={"margin": "0", "color": "#94A3B8", "fontSize": "1rem"}),
            html.Div([
                html.H2(f"{current_value:.2f}", style={"margin": "5px 0", "color": "#E2E8F0", "fontSize": "1.5rem", "display": "inline-block"}),
                html.Span(
                    f" ({percent_change:.1f}%)", 
                    style={
                        "color": "#4ADE80" if percent_change <= 0 else "#EF4444",
                        "fontSize": "0.9rem",
                        "marginLeft": "5px"
                    }
                )
            ], style={"display": "flex", "alignItems": "center"})
        ], className="card", style={"width": "200px"}),
        html.Div([
            html.H3(f"Valeur max {metric_name}", style={"margin": "0", "color": "#94A3B8", "fontSize": "1rem"}),
            html.H2(f"{max_value:.2f}", style={"margin": "5px 0", "color": "#F87171", "fontSize": "1.5rem"})
        ], className="card", style={"width": "200px"})
    ], style={"display": "flex", "gap": "15px", "marginBottom": "20px", "flexWrap": "wrap"})
    
    # Graphique principal amélioré
    fig_ts = go.Figure()
    
    # Ajouter la zone d'arrière-plan pour mettre en évidence la période normale
    time_range = olt_df["date_hour"].max() - olt_df["date_hour"].min()
    normal_threshold = avg_value * 1.2
    
    # Ajouter la ligne de référence pour la valeur normale moyenne
    fig_ts.add_hline(
        y=avg_value,
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        annotation_text="Moyenne",
        annotation_position="bottom right",
        annotation_font=dict(color="#94A3B8")
    )
    
    # Ajouter la ligne principale pour toutes les données
    fig_ts.add_trace(go.Scatter(
        x=olt_df["date_hour"],
        y=olt_df[y_col],
        mode="lines",
        name="Mesures",
        line=dict(color="#38BDF8", width=2.5),
        hovertemplate=f"{metric_name}: %{{y:.2f}}<br>Date: %{{x|%d %b %Y %H:%M}}<extra></extra>"
    ))
    
    # Ajouter des points pour les anomalies
    anomalies_ts = olt_df[olt_df[selected_anomaly_col]]
    if not anomalies_ts.empty:
        fig_ts.add_trace(go.Scatter(
            x=anomalies_ts["date_hour"],
            y=anomalies_ts[y_col],
            mode="markers",
            marker=dict(
                color="#EF4444",
                size=10,
                symbol="circle",
                line=dict(width=2, color="#FFFFFF")
            ),
            name="Anomalies",
            hovertemplate=f"{metric_name}: %{{y:.2f}}<br>Date: %{{x|%d %b %Y %H:%M}}<extra></extra>"
        ))
    
    # Ajouter une zone de seuil pour les valeurs élevées
    fig_ts.add_hline(
        y=normal_threshold,
        line=dict(color="#F87171", width=1, dash="dot"),
        annotation_text="Seuil d'alerte",
        annotation_position="top right",
        annotation_font=dict(color="#F87171")
    )
    
    fig_ts.update_layout(
        title=dict(
            text=f"Évolution de {metric_name} pour l'OLT {selected_olt}",
            font=dict(size=16, color='#60A5FA'),
            x=0.5
        ),
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="#E2E8F0",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(
            title="Date et Heure",
            showgrid=True,
            gridcolor='rgba(150, 150, 150, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title=f"Valeur de {metric_name}",
            showgrid=True,
            gridcolor='rgba(150, 150, 150, 0.1)',
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30, 41, 59, 0.8)"
        ),
        height=500
    )
    
    # Création d'un histogramme de distribution
    hist_fig = px.histogram(
        olt_df, 
        x=y_col,
        nbins=20,
        title=f"Distribution des valeurs de {metric_name}",
        template="plotly_dark",
        color_discrete_sequence=["#38BDF8"]
    )
    
    hist_fig.add_vline(
        x=avg_value,
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        annotation_text="Moyenne",
        annotation_position="top right"
    )
    
    hist_fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="#E2E8F0",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=f"Valeur de {metric_name}",
        yaxis_title="Fréquence",
        height=300,
        title=dict(
            font=dict(size=16, color='#60A5FA'),
            x=0.5
        )
    )
    
    # Création de la distribution journalière
    olt_df['hour'] = olt_df['date_hour'].dt.hour
    hour_counts = olt_df.groupby('hour')[selected_anomaly_col].mean().reset_index()
    hour_counts['hour_str'] = hour_counts['hour'].apply(lambda x: f"{x:02d}h")
    
    hour_fig = px.bar(
        hour_counts,
        x='hour_str',
        y=selected_anomaly_col,
        title="Répartition horaire des anomalies",
        template="plotly_dark",
        color_discrete_sequence=["#60A5FA"]
    )
    
    hour_fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font_color="#E2E8F0",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Heure de la journée",
        yaxis_title="Taux d'anomalies",
        height=300,
        title=dict(
            font=dict(size=16, color='#60A5FA'),
            x=0.5
        )
    )
    
    # Calcul d'un bref résumé en texto
    anomaly_perc = anomalies_count / len(olt_df) * 100 if len(olt_df) > 0 else 0
    peak_time = hour_counts.sort_values(selected_anomaly_col, ascending=False).iloc[0]['hour_str'] if not hour_counts.empty else "N/A"
    
    # Création d'un résumé statistique
    summary_card = html.Div([
        html.H3("Résumé statistique", className="section-title"),
        html.Div([
            html.P(f"• Taux d'anomalies: {anomaly_perc:.1f}% des mesures"),
            html.P(f"• Pic d'anomalies: {peak_time}"),
            html.P(f"• Valeur moyenne: {avg_value:.2f}"),
            html.P(f"• Écart maximal: {(max_value - avg_value):.2f} ({(max_value/avg_value - 1)*100:.1f}% au-dessus de la moyenne)")
        ], style={'color': '#CBD5E1'})
    ], className="card", style={"width": "100%", "marginTop": "20px"})
    
    return html.Div([
        html.H2("Visualisation OLT", className="page-title", style={"marginTop": "0"}),
        stats_cards,
        html.Div([
            dcc.Graph(figure=fig_ts, config={'displayModeBar': 'hover'}),
        ], className="card"),
        html.Div([
            html.Div([
                dcc.Graph(figure=hist_fig, config={'displayModeBar': False})
            ], className="card", style={"width": "calc(50% - 15px)"}),
            html.Div([
                dcc.Graph(figure=hour_fig, config={'displayModeBar': False})
            ], className="card", style={"width": "calc(50% - 15px)"})
        ], style={"display": "flex", "gap": "30px", "marginTop": "20px"}),
        summary_card
    ])

def create_map_content(historical_df, selected_anomaly_col, start_datetime, selected_datetime):
    historical_df["anomalie"] = historical_df[selected_anomaly_col]
    gdf_departements = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
    anomalies_by_dep = historical_df[historical_df["anomalie"]].groupby("code_departement").size().reset_index(name="nb_alertes")
    anomalies_by_dep["code_departement"] = anomalies_by_dep["code_departement"].astype(str).str.zfill(2)
    
    # Ajouter des stats pour la carte
    total_deps = anomalies_by_dep.shape[0]
    max_anomalies = anomalies_by_dep["nb_alertes"].max() if not anomalies_by_dep.empty else 0
    
    stats_cards = html.Div([
        html.Div([
            html.H2(f"{total_deps}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Départements touchés", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "230px"}),
        html.Div([
            html.H2(f"{max_anomalies}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Max anomalies dans un département", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "280px"})
    ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"})
    
    carte_df = gdf_departements.merge(anomalies_by_dep, how="left", left_on="code", right_on="code_departement")
    carte_df["nb_alertes"] = carte_df["nb_alertes"].fillna(0)
    
    # Amélioration de la carte
    fig_map = px.choropleth_mapbox(
        carte_df,
        geojson=carte_df.geometry.__geo_interface__,
        locations=carte_df.index,
        color="nb_alertes",
        hover_name="nom",
        hover_data={"nb_alertes": True, "code": True},
        mapbox_style="carto-darkmatter",
        center={"lat": 46.6, "lon": 2.6},
        zoom=4.7,
        opacity=0.7,
        color_continuous_scale="Blues",
        height=600,
        template="plotly_dark",
        labels={"nb_alertes": "Nombre d'anomalies"}
    )
    
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title="Anomalies",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside",
            tickfont=dict(color="#E2E8F0")
        ),
        paper_bgcolor="#1E293B"
    )
    
    # Tableau des top départements
    top_deps = anomalies_by_dep.sort_values("nb_alertes", ascending=False).head(10)
    top_deps = top_deps.merge(gdf_departements[["code", "nom"]], left_on="code_departement", right_on="code", how="left")
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Département", style={"textAlign": "left", "padding": "10px", "borderBottom": "1px solid #334155"}),
            html.Th("Code", style={"textAlign": "center", "padding": "10px", "borderBottom": "1px solid #334155"}),
            html.Th("Nombre d'anomalies", style={"textAlign": "right", "padding": "10px", "borderBottom": "1px solid #334155"})
        ]))
    ]
    
    table_rows = []
    for i, row in top_deps.iterrows():
        table_rows.append(html.Tr([
            html.Td(row["nom"], style={"padding": "8px", "borderBottom": "1px solid #334155"}),
            html.Td(row["code_departement"], style={"textAlign": "center", "padding": "8px", "borderBottom": "1px solid #334155"}),
            html.Td(row["nb_alertes"], style={"textAlign": "right", "padding": "8px", "borderBottom": "1px solid #334155"})
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    table = html.Div([
        html.H3("Top 10 des départements", className="section-title"),
        html.Table(table_header + table_body, style={
            "width": "100%",
            "borderCollapse": "collapse",
            "marginTop": "10px"
        })
    ], className="card", style={"marginTop": "20px"})
    
    return html.Div([
        html.H2("Carte des anomalies par département", className="page-title", style={"marginTop": "0"}),
        stats_cards,
        html.Div([
            dcc.Graph(figure=fig_map, style={'width': '100%', 'height': '600px'})
        ], className="card"),
        table
    ])

def create_journal_content(journal_df, selected_anomaly_col):
    """Fonction améliorée pour le journal des anomalies"""
    # Préparer les colonnes à afficher (simplifier l'affichage)
    display_columns = [
        "date_hour", "olt_name", "boucle", "type_olt_model", 
        "code_departement", "avg_dns_time", "avg_latence_scoring", 
        "avg_score_scoring", selected_anomaly_col
    ]
    
    journal_df = journal_df[display_columns].copy()
    
    # Renommer les colonnes pour un affichage plus lisible
    column_renames = {
        "date_hour": "Date et Heure",
        "olt_name": "OLT",
        "boucle": "Boucle",
        "type_olt_model": "Type OLT",
        "code_departement": "Département",
        "avg_dns_time": "DNS Moyen",
        "avg_latence_scoring": "Latence Moyenne",
        "avg_score_scoring": "Score Moyen",
        selected_anomaly_col: "Anomalie"
    }
    
    journal_df = journal_df.rename(columns=column_renames)
    
    # Formatter les valeurs numériques pour un meilleur affichage
    for col in ["DNS Moyen", "Latence Moyenne", "Score Moyen"]:
        if col in journal_df.columns:
            journal_df[col] = journal_df[col].round(2)
    
    # Convertir les booléens d'anomalies en texte
    journal_df["Anomalie"] = journal_df["Anomalie"].map({True: "Oui", False: "Non"})
    
    # Formatter les dates
    journal_df["Date et Heure"] = journal_df["Date et Heure"].dt.strftime("%Y-%m-%d %H:%M")
    
    # Préparer les colonnes pour DataTable
    table_columns = [
        {"name": col, "id": col, "selectable": True} for col in journal_df.columns
    ]
    
    # Préparer les statistiques de synthèse
    total_anomalies = len(journal_df)
    unique_olts = journal_df["OLT"].nunique()
    unique_boucles = journal_df["Boucle"].nunique()
    unique_depts = journal_df["Département"].nunique()
    
    stats_cards = html.Div([
        html.Div([
            html.H2(f"{total_anomalies}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Anomalies", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "180px"}),
        html.Div([
            html.H2(f"{unique_olts}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("OLTs uniques", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "180px"}),
        html.Div([
            html.H2(f"{unique_boucles}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Boucles uniques", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "180px"}),
        html.Div([
            html.H2(f"{unique_depts}", style={"margin": "0", "color": "#38BDF8", "fontSize": "2.5rem"}),
            html.P("Départements", style={"margin": "0", "color": "#94A3B8"})
        ], className="card", style={"textAlign": "center", "width": "180px"})
    ], style={"display": "flex", "gap": "15px", "marginBottom": "20px", "flexWrap": "wrap"})
    
    # Préparer le CSV pour export
    buffer = io.StringIO()
    journal_df.to_csv(buffer, index=False)
    csv_data = base64
    # Bouton d'export amélioré
    download_link = html.A(
        html.Div([
            html.I(className="fas fa-download", style={"marginRight": "8px"}),
            "Exporter en CSV"
        ], style={"display": "flex", "alignItems": "center"}),
        href=f"data:text/csv;base64,{csv_data}",
        download="journal_anomalies.csv",
        className="export-link",
        style={"marginTop": "15px", "display": "inline-block"}
    )
    
    return html.Div([
        html.H2("Journal des anomalies", className="page-title", style={"marginTop": "0"}),
        stats_cards,
        html.Div([
            dash.dash_table.DataTable(
                data=journal_df.to_dict("records"),
                columns=table_columns,
                style_table={'overflowX': 'auto'},
                style_data={
                    'backgroundColor': '#1E293B',
                    'color': '#E2E8F0',
                    'border': '1px solid #334155'
                },
                style_header={
                    'backgroundColor': '#0F172A',
                    'color': '#38BDF8',
                    'fontWeight': 'bold',
                    'border': '1px solid #334155',
                    'textAlign': 'left'
                },
                style_cell={
                    'padding': '10px',
                    'fontFamily': 'Inter, sans-serif',
                    'textAlign': 'left'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Anomalie', 'filter_query': '{Anomalie} eq "Oui"'},
                        'backgroundColor': 'rgba(239, 68, 68, 0.2)',
                        'color': '#EF4444'
                    }
                ],
                page_size=15,
                page_action="native",
                sort_action="native",
                filter_action="native",
                export_format="csv",
                row_selectable="multi",
                selected_rows=[],
                style_as_list_view=True
            )
        ], className="card"),
        download_link
    ])

# Configuration de l'application Dash
app = dash.Dash(__name__)
app.title = "Anomalie Network Dashboard"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        <style>
            html, body {
                height: 100%;
                margin: 0;
                background-color: #0F172A;
                color: #E2E8F0;
                font-family: 'Inter', sans-serif;
            }
            /* Conteneur global en flex column pour un footer sticky */
            .page-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            header {
                background-color: #1E293B;
                padding: 15px 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #334155;
            }
            .header-left {
                text-align: left;
            }
            .title-row {
                display: flex;
                align-items: center;
            }
            .title {
                margin: 0;
                font-size: 2.2em;
                font-weight: 800;
                color: #60A5FA;
                letter-spacing: -0.5px;
                text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
            }
            .nexialog-logo {
                max-height: 40px;
                margin-left: 20px;
                filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.3));
            }
            .subtitle {
                margin: 2px 0 0 0;
                font-size: 1em;
                color: #94A3B8;
                font-weight: 300;
            }
            .header-right {
                text-align: right;
            }
            
            /* Style des onglets amélioré */
            .nav-bar {
                display: flex;
                gap: 5px;
                justify-content: flex-end;
            }
            .nav-bar a {
                color: #CBD5E1;
                text-decoration: none;
                font-size: 1.05em;
                font-weight: 600;
                transition: all 0.2s;
                padding: 10px 20px;
                border-radius: 8px 8px 0 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                position: relative;
            }
            /* Effet au survol */
            .nav-bar a:hover {
                color: #38BDF8;
                background-color: rgba(15, 23, 42, 0.3);
            }
            /* Style pour l'onglet actif - similaire à votre capture d'écran */
            .nav-bar a.active {
                color: #38BDF8;
                background-color: #0F172A;
                border-bottom: 3px solid #38BDF8;
            }
            
            main {
                flex: 1;
                padding: 30px;
            }
            footer {
                background-color: #1E293B;
                padding: 20px;
                text-align: center;
                color: #94A3B8;
                border-top: 1px solid #334155;
            }
            footer .partners {
                margin-bottom: 15px;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
            }
            footer .partners strong {
                margin-right: 10px;
                font-weight: 500;
            }
            footer .partners img {
                max-height: 30px;
                transition: transform 0.3s ease;
                filter: grayscale(50%);
            }
            footer .partners img:hover {
                filter: grayscale(0%);
                transform: scale(1.05);
            }
            footer .git-profiles {
                margin-top: 15px;
            }
            footer .git-profiles a {
                color: #60A5FA;
                text-decoration: none;
                margin: 0 10px;
                transition: color 0.3s;
            }
            footer .git-profiles a:hover {
                color: #38BDF8;
                text-decoration: underline;
            }
            
            /* Styles pour éléments spécifiques */
            .highlight {
                color: #60A5FA;
                font-weight: 600;
                margin-bottom: 12px;
                margin-top: 20px;
            }
            /* AMÉLIORATION TITRES */
            .page-title {
                color: #60A5FA;
                font-size: 1.8em;
                font-weight: 700;
                margin-top: 0;
                margin-bottom: 20px;
                border-bottom: 2px solid #334155;
                padding-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .section-title {
                color: #60A5FA;
                font-size: 1.3em;
                font-weight: 600;
                margin-top: 20px;
                margin-bottom: 15px;
                padding-bottom: 5px;
                border-bottom: 1px solid #334155;
            }
            
            .card {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 16px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* AMÉLIORATION CARTES STATISTIQUES */
            .card h2 {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                color: #38BDF8;
                text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            }
            .card h3 {
                color: #94A3B8;
                font-size: 1rem;
                font-weight: 500;
                margin: 0;
            }
            .card p {
                color: #CBD5E1;
            }
            
            /* Styles pour les contrôles */
            .Select-control, .Select-menu-outer {
                background-color: #0F172A !important;
                color: #E2E8F0 !important;
                border: 1px solid #475569 !important;
                border-radius: 6px !important;
            }
            .Select-value-label {
                color: #E2E8F0 !important;
            }
            .Select-menu-outer {
                background-color: #0F172A !important;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
            }
            .Select-option {
                background-color: #0F172A !important;
                color: #E2E8F0 !important;
            }
            .Select-option.is-focused {
                background-color: #2D3D50 !important;
            }
            .DateInput, .SingleDatePickerInput {
                background-color: #0F172A !important;
                border: 1px solid #475569 !important;
                border-radius: 6px !important;
            }
            .DateInput_input {
                color: #E2E8F0 !important;
                background-color: #0F172A !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            /* AMÉLIORATION BOUTONS RADIO */
            input[type="radio"] {
                accent-color: #38BDF8;
                width: 18px;
                height: 18px;
                margin-right: 8px;
            }
            .radio-label {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
                color: #CBD5E1;
                font-weight: 500;
                cursor: pointer;
            }
        </style>
    </head>
    <style>
    /* Amélioration du volet de filtres */
    .left-panel-title {
        color: #60A5FA;
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-bottom: 2px solid #38BDF8;
        padding-bottom: 8px;
    }
    
    /* Catégories de filtres */
    .filter-category {
        color: #38BDF8;
        font-size: 1.2em;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    
    /* Amélioration radio buttons et labels */
    .radio-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        transition: transform 0.2s;
        cursor: pointer;
    }
    
    .radio-item:hover {
        transform: translateX(5px);
    }
    
    .radio-item input[type="radio"] {
        width: 18px;
        height: 18px;
        margin-right: 10px;
        cursor: pointer;
        accent-color: #60A5FA;
    }
    
    .radio-item label {
        color: #CBD5E1;
        font-weight: 500;
        cursor: pointer;
    }
    
    /* Style spécial pour l'option sélectionnée */
    .radio-item input[type="radio"]:checked + label {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Ajouter un fond distinct au volet gauche */
    .filter-panel {
        background-color: #111827;
        border-radius: 10px;
        border: 1px solid #1E293B;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    
    /* Style pour les dropdowns */
    .filter-dropdown {
        margin-bottom: 15px;
    }
    
    .filter-dropdown-label {
        color: #CBD5E1;
        font-weight: 500;
        margin-bottom: 8px;
        display: block;
    }
    
    /* Amélioration des éléments de date */
    .date-picker-container {
        margin-bottom: 20px;
    }
    
    .date-help-text {
        color: #94A3B8;
        font-size: 0.85em;
        font-style: italic;
        margin-top: 5px;
    }
    
    /* Ajouter une bordure distinctive sur les entrées */
    .DateInput {
        border: 1px solid #38BDF8 !important;
    }
    
    label {
        display: block;
        margin-bottom: 8px;
        font-weight: 500;
        color: #CBD5E1;
    }
    .export-link {
        display: inline-flex;
        align-items: center;
        margin-top: 15px;
        padding: 10px 18px;
        background-color: #38BDF8;
        color: #0F172A;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .export-link:hover {
        background-color: #0EA5E9;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    
    /* AMÉLIORATION ZONE DE FILTRES */
    .filter-section {
        background-color: #111827;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    .filter-section h3 {
        color: #60A5FA;
        font-size: 1.2em;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 15px;
        border-bottom: 2px solid #334155;
        padding-bottom: 8px;
        text-transform: uppercase;
    }
    
    /* Styles pour améliorer la sélection OLT */
    
    /* Zone principale de visualisation avec un appel à l'action */
    .visualization-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: #0F172A;
        border-radius: 10px;
        border: 1px solid #334155;
        padding: 40px;
        text-align: center;
        height: 400px;
        position: relative;
        overflow: hidden;
    }
    
    .visualization-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(56, 189, 248, 0.05) 0%, rgba(15, 23, 42, 0) 70%);
        z-index: 0;
    }
    
    .selection-prompt {
        position: relative;
        z-index: 1;
    }
    
    .selection-title {
        color: #60A5FA;
        font-size: 1.8em;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    
    .selection-description {
        color: #94A3B8;
        font-size: 1.1em;
        margin-bottom: 30px;
        max-width: 600px;
    }
    
    /* Bouton pour diriger l'attention vers le sélecteur */
    .selection-button {
        background-color: #38BDF8;
        color: #0F172A;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .selection-button:hover {
        background-color: #0EA5E9;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    
    /* Animation pulsante pour attirer l'attention vers le menu de sélection */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(56, 189, 248, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0);
        }
    }
    
    /* Améliorations pour le dropdown de sélection OLT */
    .olt-selection-dropdown {
        width: 100%;
        animation: pulse 2s infinite;
    }
    
    .olt-selection-container {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #38BDF8;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.15);
    }
    
    .highlight-selector {
        animation: pulse 2s infinite;
    }
    
    .olt-selection-title {
        color: #38BDF8;
        font-size: 1.3em;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .olt-category {
        color: #CBD5E1;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Indications visuelles pour les OLTs disponibles */
    .olt-count {
        display: inline-block;
        background-color: #38BDF8;
        color: #0F172A;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    
    /* Tooltip d'aide pour guider l'utilisateur */
    .tooltip-container {
        position: relative;
        display: inline-block;
        margin-left: 10px;
    }
    
    .tooltip-icon {
        color: #60A5FA;
        cursor: pointer;
        font-size: 0.9em;
    }
    
    .tooltip-text {
        visibility: hidden;
        width: 250px;
        background-color: #1E293B;
        color: #CBD5E1;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        font-size: 0.9em;
        font-weight: normal;
        text-transform: none;
        letter-spacing: normal;
    }
    
    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Table header improvements */
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
        background-color: #0F172A !important;
        color: #60A5FA !important;
        font-weight: 700 !important;
        padding: 12px 10px !important;
        border-bottom: 2px solid #334155 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Responsive improvements */
    @media (max-width: 1200px) {
        .nav-bar {
            gap: 10px;
        }
        
        .nav-bar a {
            font-size: 0.8em;
            padding: 8px 10px;
        }
    }
    @media (max-width: 768px) {
        .nav-bar {
            flex-wrap: wrap;
            justify-content: center;
        }
    }
</style>
    <div class="page-container">
            <header>
                <div class="header-left">
                    <div class="title-row">
                        <h1 class="title">AnomalyX</h1>
                        <img src="/assets/nexialog_logo.png" alt="Logo Nexialog" class="nexialog-logo">
                    </div>
                    <p class="subtitle">Making Network Troubleshooting Painless</p>
                </div>
                <div class="header-right">
                    <nav class="nav-bar">
                        <a href="/dashboard" id="tab-dashboard">ALERTES</a>
                        <a href="/journal" id="tab-journal">JOURNAL</a>
                        <a href="/map" id="tab-map">CARTE</a>
                        <a href="/visual" id="tab-visual">VISUALISATION</a>
                        <a href="/scenario-builder" id="tab-scenario">SCENARIO</a>
                        <a href="/actions" id="tab-actions">ACTIONS</a>
                    </nav>
                </div>
            </header>
            <main>
                {%app_entry%}
            </main>
            <footer>
                <div class="partners">
                    <strong>Partenaires :</strong>
                    <img src="/assets/sfr_logo.png" alt="Logo SFR">
                    <img src="/assets/mosef_logo.png" alt="Logo Mosef">
                </div>
                <div class="git-profiles">
                    <a href="https://github.com/lucasvazelle" target="_blank">lucasvazelle</a> |
                    <a href="https://github.com/Ayamokht" target="_blank">Ayamokht</a> |
                    <a href="https://github.com/lazstx" target="_blank">lazstx</a> |
                    <a href="https://github.com/hmnthy" target="_blank">hmnthy</a>
                </div>
                {%config%}
                {%scripts%}
                {%renderer%}
                <script>
                    // Script pour gérer les onglets actifs
                    function updateActiveTab() {
                        // Obtenir le chemin actuel
                        const pathname = window.location.pathname || '/dashboard';
                        
                        // Retirer la classe active de tous les onglets
                        document.querySelectorAll('.nav-bar a').forEach(tab => {
                            tab.classList.remove('active');
                        });
                        
                        // Trouver l'onglet qui correspond au chemin actuel
                        let tabId = '';
                        
                        // Gérer le cas de la route par défaut
                        if (pathname === '/' || pathname === '') {
                            tabId = 'tab-dashboard';
                        } else {
                            // Extraire l'id de l'onglet à partir du chemin
                            const route = pathname.split('/')[1];
                            tabId = route ? `tab-${route}` : 'tab-dashboard';
                        }
                        
                        // Ajouter la classe active à l'onglet actuel
                        const activeTab = document.getElementById(tabId);
                        if (activeTab) {
                            activeTab.classList.add('active');
                        }
                    }

                    // Exécuter cette fonction au chargement de la page
                    document.addEventListener('DOMContentLoaded', function() {
                        updateActiveTab();
                        
                        // Observer les changements dans le contenu pour mettre à jour l'onglet actif
                        const observer = new MutationObserver(function(mutations) {
                            updateActiveTab();
                        });
                        
                        // Observer les changements dans la zone de contenu principale
                        const mainContent = document.getElementById('page-content');
                        if (mainContent) {
                            observer.observe(mainContent, { childList: true });
                        }
                        
                        // Gérer le click sur le bouton de sélection OLT
                        document.addEventListener('click', function(e) {
                            if (e.target.id === 'select-olt-btn' || e.target.closest('#select-olt-btn')) {
                                const oltSelector = document.getElementById('olt-visual-controls');
                                if (oltSelector) {
                                    oltSelector.classList.add('highlight-selector');
                                    // Scroller vers le sélecteur OLT
                                    oltSelector.scrollIntoView({ behavior: 'smooth' });
                                    
                                    // Retirer l'effet de highlight après quelques secondes
                                    setTimeout(function() {
                                        oltSelector.classList.remove('highlight-selector');
                                    }, 3000);
                                }
                            }
                        });
                        
                        // Amélioration des boutons radio pour la visualisation
                        document.querySelectorAll('input[type="radio"]').forEach(function(radio) {
                            // Trouver le label associé au radio
                            const label = document.querySelector(`label[for="${radio.id}"]`) || 
                                         radio.nextElementSibling;
                                         
                            if (label) {
                                // Créer un wrapper div
                                const wrapper = document.createElement('div');
                                wrapper.className = 'radio-item';
                                
                                // Placer le radio et son label dans le wrapper
                                const radioParent = radio.parentNode;
                                radioParent.insertBefore(wrapper, radio);
                                wrapper.appendChild(radio);
                                wrapper.appendChild(label);
                                
                                // Ajouter un event listener pour highlight
                                radio.addEventListener('change', function() {
                                    // Retirer highlight de tous les radios du même groupe
                                    document.querySelectorAll(`input[name="${radio.name}"]`).forEach(function(r) {
                                        const rWrapper = r.closest('.radio-item');
                                        if (rWrapper) {
                                            rWrapper.classList.remove('selected');
                                        }
                                    });
                                    
                                    // Ajouter highlight au radio sélectionné
                                    wrapper.classList.add('selected');
                                });
                                
                                // Appliquer highlight initial si déjà sélectionné
                                if (radio.checked) {
                                    wrapper.classList.add('selected');
                                }
                            }
                        });
                    });
                </script>
            </footer>
        </div>
    </body>
</html>
'''

# Maintenant, modifiez votre app.layout pour utiliser les nouvelles classes :

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        # Colonne des filtres avec sélection OLT intégrée
        html.Div([
            html.Div([
                html.H3("FILTRES", className="left-panel-title", style={"marginTop": "0"}),
                html.H4("Date", className="filter-category"),
                html.Div([
                    dcc.DatePickerSingle(
                        id='date-picker',
                        min_date_allowed=min_date.date(),
                        max_date_allowed=max_date.date(),
                        date=max_date.date(),
                        display_format='YYYY-MM-DD',
                        with_portal=True,
                        first_day_of_week=1
                    ),
                    html.Small(
                        "Données disponibles du 3 déc. 2024 au 31 jan. 2025",
                        className="date-help-text"
                    )
                ], className="date-picker-container"),
                
                html.H4("Heure", className="filter-category"),
                html.Div([
                    dcc.Dropdown(
                        id='hour-picker',
                        options=[{"label": f"{h:02d}:00", "value": h} for h in range(24)],
                        value=datetime.now().hour,
                        clearable=False
                    )
                ], className="filter-dropdown"),
                
                html.H4("Période d'analyse", className="filter-category"),
                dcc.RadioItems(
                    id='hours-back',
                    options=[
                        {"label": "Heure", "value": 1},
                        {"label": "Journée", "value": 24},
                        {"label": "Semaine", "value": 168},
                        {"label": "Mois", "value": 720}
                    ],
                    value=24,
                    className="radio-group",
                    inputClassName="radio-input",
                    labelClassName="radio-label",
                    labelStyle={'display': 'inline-block'}
                ),
                
                html.H4("Type d'anomalie", className="filter-category"),
                dcc.RadioItems(
                    id='anomaly-type',
                    options=[{"label": k, "value": k} for k in anomaly_col_map.keys()],
                    value="DNS",
                    className="radio-group",
                    inputClassName="radio-input",
                    labelClassName="radio-label",
                    labelStyle={'display': 'inline-block'}
                ),
            ], className="filter-panel"),
            
            # Filtres supplémentaires pour le Journal (conserver cette partie mais ajouter les classes)
            html.Div([
                html.H3("Filtres avancés", className="left-panel-title", style={"marginTop": "20px"}),
                html.H4("Filtrer par Boucle", className="filter-category"),
                dcc.Dropdown(
                    id='boucle-filter',
                    options=boucle_options,
                    placeholder="Sélectionnez une Boucle",
                    multi=True,
                    className="filter-dropdown"
                ),
                html.H4("Filtrer par Type OLT", className="filter-category"),
                dcc.Dropdown(
                    id='olt-type-filter',
                    options=olt_type_options,
                    placeholder="Sélectionnez un Type OLT",
                    multi=True,
                    className="filter-dropdown"
                )
            ], id="journal-filters", style={"display": "none"}, className="filter-panel"),
            
            # Bloc pour le sélecteur OLT (affiché uniquement sur la page Visualisation)
            html.Div([
                html.H3("Sélection OLT", className="left-panel-title", style={"marginTop": "20px"}),
                html.H4("OLT", className="filter-category"),
                dcc.Dropdown(
                    id='olt-picker',
                    options=[],
                    placeholder="Sélectionnez un OLT",
                    className="filter-dropdown"
                )
            ], id="olt-visual-controls", style={"display": "none"}, className="filter-panel")
        ], style={'width': '320px', 'float': 'left'}),
        
        # Colonne du contenu principal
        html.Div(id="page-content", style={
            'marginLeft': '350px', 
            'padding': '20px',
            'backgroundColor': '#1E293B',
            'borderRadius': '8px',
            'minHeight': 'calc(100vh - 250px)',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'border': '1px solid #334155'
        })
    ])
])

# Callback pour afficher/cacher les filtres supplémentaires pour le Journal
@app.callback(
    Output('journal-filters', 'style'),
    Input('url', 'pathname')
)
def toggle_journal_filters(pathname):
    if pathname == '/journal':
        return {"display": "block", "marginTop": "20px"}
    return {"display": "none"}

# Callback pour mettre à jour le sélecteur OLT (affiché uniquement sur la page Visualisation)
@app.callback(
    Output('olt-visual-controls', 'style'),
    Output('olt-picker', 'options'),
    Input('url', 'pathname'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('anomaly-type', 'value')
)
def update_olt_selector(pathname, date_str, hour, anomaly_type):
    if pathname != '/visual':
        return {"display": "none"}, []
    selected_datetime = datetime.combine(pd.to_datetime(date_str).date(), time(int(hour)))
    df_hour = df[(df["date_hour"] >= selected_datetime) &
                 (df["date_hour"] < selected_datetime + timedelta(hours=1))]
    df_hour = df_hour[df_hour[anomaly_col_map[anomaly_type]]]
    olts = sorted(df_hour["olt_name"].unique())
    return {"display": "block", "marginTop": "10px"}, [{'label': o, 'value': o} for o in olts]

# Callback pour afficher le contenu en fonction de l'URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('hours-back', 'value'),
    Input('anomaly-type', 'value'),
    Input('olt-picker', 'value'),
    Input('boucle-filter', 'value'),
    Input('olt-type-filter', 'value')
)



def display_page(pathname, date_str, hour, hours_back, anomaly_type, selected_olt, boucle_filter, olt_type_filter):
    if pathname not in ['/dashboard', '/journal', '/map', '/visual', '/scenario-builder', '/actions']:
        pathname = '/dashboard'
    selected_datetime = datetime.combine(pd.to_datetime(date_str).date(), time(int(hour)))
    start_datetime = selected_datetime - timedelta(hours=int(hours_back))
    selected_anomaly_col = anomaly_col_map[anomaly_type]

    if pathname == '/dashboard':
        historical_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] <= selected_datetime)].copy()
        historical_df["anomalie"] = historical_df[selected_anomaly_col]
        attrs = ["code_departement", "type_olt_model", "type_boucle", "boucle"]
        pie_charts = []
        for attr in attrs:
            counts = historical_df[historical_df["anomalie"]].groupby(attr).size().reset_index(name="nb_alertes")
            if attr in ["code_departement", "boucle"]:
                counts = counts.sort_values("nb_alertes", ascending=False).head(3)
            else:
                counts = counts.sort_values("nb_alertes", ascending=False).head(5)
            pie_fig = px.pie(
                counts,
                names=attr,
                values="nb_alertes",
                hole=0.4,
                title=f"{attr.replace('_', ' ').title()}",
                template="plotly_dark"
            )
            pie_fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(font_color='#e0e0e0'),
                font=dict(size=10)
            )
            pie_charts.append(html.Div(
                dcc.Graph(figure=pie_fig, config={'displayModeBar': False}),
                style={'width': '300px', 'margin': '10px'}
            ))
        volume_details = []
        for attr in attrs:
            df_attr = historical_df[historical_df["anomalie"]]
            total_volume = df_attr.shape[0]
            limit = 3 if attr in ["code_departement", "boucle"] else 5
            counts = df_attr.groupby(attr).size().reset_index(name="nb_alertes")
            top_items = counts.sort_values("nb_alertes", ascending=False).head(limit)
            top_list = html.Ul([html.Li(f"{row[attr]} : {row['nb_alertes']} alertes")
                                for _, row in top_items.iterrows()])
            volume_details.append(html.Div([
                html.H4(attr.replace('_', ' ').title(), className="highlight"),
                html.P(f"Volume total d'anomalies: {total_volume}", style={'fontSize': '12px'}),
                top_list
            ], className="card", style={'width': '300px'}))
        return html.Div([
            html.H3("Détails des volumes d'anomalies", className="highlight"),
            html.Div(volume_details, style={"display": "flex", "flexWrap": "wrap"}),
            html.H3("Répartition des anomalies", className="highlight"),
            html.Div(pie_charts, style={"display": "flex", "flexWrap": "wrap"})
        ])
    
        # Dans le callback display_page, modifiez la section concernant '/journal'
    elif pathname == '/journal':
        journal_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] < selected_datetime)]
        journal_df = journal_df[journal_df[selected_anomaly_col]]
        if boucle_filter:
            journal_df = journal_df[journal_df["boucle"].isin(boucle_filter)]
        if olt_type_filter:
            journal_df = journal_df[journal_df["type_olt_model"].isin(olt_type_filter)]
        journal_df = journal_df.sort_values("date_hour", ascending=False)
        
        # Utiliser la fonction améliorée pour créer le journal
        return create_journal_content(journal_df, selected_anomaly_col)
    
    elif pathname == '/map':
        historical_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] <= selected_datetime)].copy()
        historical_df["anomalie"] = historical_df[selected_anomaly_col]
        gdf_departements = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")
        anomalies_by_dep = historical_df[historical_df["anomalie"]].groupby("code_departement").size().reset_index(name="nb_alertes")
        anomalies_by_dep["code_departement"] = anomalies_by_dep["code_departement"].astype(str).str.zfill(2)
        carte_df = gdf_departements.merge(anomalies_by_dep, how="left", left_on="code", right_on="code_departement")
        carte_df["nb_alertes"] = carte_df["nb_alertes"].fillna(0)
        fig_map = px.choropleth_mapbox(
            carte_df,
            geojson=carte_df.geometry.__geo_interface__,
            locations=carte_df.index,
            color="nb_alertes",
            hover_name="nom",
            mapbox_style="carto-darkmatter",
            center={"lat": 46.6, "lon": 2.6},
            zoom=4.5,
            opacity=0.6,
            color_continuous_scale="YlOrRd",
            height=600,
            template="plotly_dark",
            labels={"nb_alertes": "Nombre d'anomalies"}
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return html.Div([
            html.H3("Carte des anomalies", className="highlight"),
            dcc.Graph(figure=fig_map, style={'width': '100%', 'height': '600px'})
        ])
    
    elif pathname == '/visual':
        if not selected_olt:
            return html.Div([html.H4("Visualisation", className="highlight"), html.P("Veuillez sélectionner un OLT")])
        visual_start = selected_datetime - timedelta(hours=int(hours_back))
        olt_df = df[(df["olt_name"] == selected_olt) & (df["date_hour"] >= visual_start) & (df["date_hour"] <= selected_datetime)].sort_values("date_hour")
        
        # Utiliser la fonction avancée pour la visualisation
        return create_visual_content(olt_df, selected_olt, selected_anomaly_col)

        
    elif pathname == '/scenario-builder':
        
        return html.Div([
            html.H3("Scenario Builder", className="highlight"),
            html.P("Ici vous pourrez créer et sauvegarder des scénarios de simulation d'anomalies. (Fonctionnalités à implémenter)")
        ])
        
    elif pathname == '/actions':
        return html.Div([
            html.H3("Actions", className="highlight"),
            html.P("Cette page permettra de lancer des actions correctives ou des scripts automatisés suite à des alertes. (Fonctionnalités à implémenter)")
        ])
    return html.Div(["Page non reconnue"])



if __name__ == '__main__':
    app.run(debug=True)
