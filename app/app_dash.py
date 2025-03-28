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

# Chargement des données
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

# Options pour les filtres du Journal (Boucle et Type OLT)
boucle_options = [{'label': str(val), 'value': str(val)}
                  for val in sorted(df["boucle"].dropna().unique())]
olt_type_options = [{'label': str(val), 'value': str(val)}
                    for val in sorted(df["type_olt_model"].dropna().unique())]

# Dates limites
min_date = datetime(2024, 12, 3)
max_date = datetime(2025, 1, 31)

# Mapping pour les types d'anomalies
anomaly_col_map = {
    "DNS": "is_jump_avg_dns_time",
    "Latence Scoring": "is_jump_avg_latence_scoring",
    "Score Scoring": "is_jump_avg_score_scoring"
}

# Configuration de l'application Dash
app = dash.Dash(__name__)
app.title = "Anomalie Network Dashboard"

# Nouvelle index_string avec header et footer intégrés, dans un thème sombre aux accents Nexialog
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;700&display=swap" rel="stylesheet">
        <style>
            html, body {
                height: 100%;
                margin: 0;
            }
            /* Conteneur global pour footer sticky */
            .page-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            header {
                background-color: #1c1c1c;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.5);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
            }
            .header-left {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }
            .title-row {
                display: flex;
                align-items: center;
            }
            .title {
                margin: 0;
                font-size: 2em;
                color: #0072CE;
            }
            .nexialog-logo {
                max-height: 50px;
                margin-left: 20px;
            }
            .subtitle {
                margin: 5px 0 0 0;
                font-size: 1em;
                color: #cccccc;
            }
            .header-right {
                text-align: right;
                margin-top: 10px;
            }
            .nav-bar {
                display: flex;
                gap: 30px;
                justify-content: flex-end;
            }
            .nav-bar a {
                color: #e0e0e0;
                text-decoration: none;
                font-size: 1.2em;
                transition: color 0.3s;
            }
            .nav-bar a:hover {
                color: #0072CE;
            }
            main {
                flex: 1;
            }
            /* Footer horizontal */
            footer {
                background-color: #1c1c1c;
                padding: 20px;
                text-align: center;
                color: #cccccc;
            }
            footer .partners {
                margin-bottom: 10px;
            }
            footer .partners img {
                max-height: 40px;
                margin: 0 10px;
            }
            footer .git-profiles a {
                color: #0072CE;
                text-decoration: none;
                margin: 0 10px;
            }
            footer .git-profiles a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="page-container">
            <header>
                <div class="header-left">
                    <div class="title-row">
                        <h1 class="title">NetworkLog</h1>
                        <img src="/assets/nexialog_logo.png" alt="Logo Nexialog" class="nexialog-logo">
                    </div>
                    <p class="subtitle">Une solution de dédection d'anomalie réseau</p>
                </div>
                <div class="header-right">
                    <nav class="nav-bar">
                        <a href="/dashboard" id="tab-dashboard">Alertes</a>
                        <a href="/journal" id="tab-journal">Journal</a>
                        <a href="/map" id="tab-map">Carte</a>
                        <a href="/visual" id="tab-visual">Visualisation</a>
                        <a href="/scenario-builder" id="tab-scenario">Scenario Builder</a>
                        <a href="/actions" id="tab-actions">Actions</a>
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
            </footer>
        </div>
    </body>
</html>
'''

# Layout principal réparti en deux colonnes : sélecteur (filtres) à gauche, contenu principal à droite
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        # Colonne des filtres (sélecteur)
        html.Div([
            html.Label("Date", className="highlight"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=min_date.date(),
                max_date_allowed=max_date.date(),
                date=max_date.date(),
                display_format='YYYY-MM-DD'
            ),
            html.Br(),
            html.Label("Heure", className="highlight"),
            dcc.Dropdown(
                id='hour-picker',
                options=[{"label": f"{h:02d}:00", "value": h} for h in range(24)],
                value=datetime.now().hour,
                clearable=False,
                style={'color': '#0072CE', 'backgroundColor': '#1e1e1e'}
            ),
            html.Br(),
            html.Label("Période d'analyse", className="highlight"),
            dcc.RadioItems(
                id='hours-back',
                options=[
                    {"label": "Heure", "value": 1},
                    {"label": "Journée", "value": 24},
                    {"label": "Semaine", "value": 168},
                    {"label": "Mois", "value": 720}
                ],
                value=24,
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Br(),
            html.Label("Type d'anomalie", className="highlight"),
            dcc.RadioItems(
                id='anomaly-type',
                options=[{"label": k, "value": k} for k in anomaly_col_map.keys()],
                value="DNS",
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Br(),
            # Filtres supplémentaires pour le Journal
            html.Div([
                html.Label("Filtrer par Boucle", className="highlight"),
                dcc.Dropdown(
                    id='boucle-filter',
                    options=boucle_options,
                    placeholder="Sélectionnez une Boucle",
                    multi=True,
                    style={'color': '#0072CE', 'backgroundColor': '#1e1e1e'}
                ),
                html.Br(),
                html.Label("Filtrer par Type OLT", className="highlight"),
                dcc.Dropdown(
                    id='olt-type-filter',
                    options=olt_type_options,
                    placeholder="Sélectionnez un Type OLT",
                    multi=True,
                    style={'color': '#0072CE', 'backgroundColor': '#1e1e1e'}
                )
            ], id="journal-filters", style={"display": "none", "marginTop": "20px"}),
            html.Br(),
            # Bloc pour le sélecteur OLT (affiché uniquement sur la page Visualisation)
            html.Div([
                html.Label("OLT", className="highlight"),
                dcc.Dropdown(
                    id='olt-picker',
                    options=[],
                    placeholder="Sélectionnez un OLT",
                    style={'color': '#0072CE', 'backgroundColor': '#1e1e1e'}
                )
            ], id="olt-visual-controls", style={"display": "none", "marginTop": "10px"})
        ], style={'width': '300px', 'float': 'left', 'padding': '10px', 'marginLeft': '20px'}),
        # Colonne du contenu principal (fixe)
        html.Div(id="page-content", style={'marginLeft': '340px', 'padding': '10px'})
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
    elif pathname == '/journal':
        journal_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] < selected_datetime)]
        journal_df = journal_df[journal_df[selected_anomaly_col]]
        if boucle_filter:
            journal_df = journal_df[journal_df["boucle"].isin(boucle_filter)]
        if olt_type_filter:
            journal_df = journal_df[journal_df["type_olt_model"].isin(olt_type_filter)]
        journal_df = journal_df.sort_values("date_hour", ascending=False)
        friendly_columns = [{"name": col.replace("is_jump", "anomalie").replace("_", " ").title(), "id": col}
                            for col in journal_df.columns]
        buffer = io.StringIO()
        journal_df.to_csv(buffer, index=False)
        csv_data = base64.b64encode(buffer.getvalue().encode()).decode()
        download_link = html.A(
            "Exporter en CSV",
            href=f"data:text/csv;base64,{csv_data}",
            download="journal_anomalies.csv",
            className="export-link"
        )
        return html.Div([
            html.H3("Journal des anomalies", className="highlight"),
            dash.dash_table.DataTable(
                data=journal_df.to_dict("records"),
                columns=friendly_columns,
                style_table={'overflowY': 'auto', 'height': 'calc(100vh - 200px)'},
                page_size=20,
                filter_action="native",
                sort_action="native",
                style_header={'backgroundColor': '#2a2a2a', 'color': '#e0e0e0', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#1e1e1e', 'color': '#e0e0e0'}
            ),
            download_link
        ])
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
        y_col = "avg_dns_time"
        if selected_anomaly_col == "is_jump_avg_latence_scoring":
            y_col = "avg_latence_scoring"
        elif selected_anomaly_col == "is_jump_avg_score_scoring":
            y_col = "avg_score_scoring"
        fig_ts = px.line(
            olt_df,
            x="date_hour",
            y=y_col,
            title="Moyenne de la métrique sur la période pour l'OLT choisi",
            markers=True,
            template="plotly_dark"
        )
        anomalies_ts = olt_df[olt_df[selected_anomaly_col]]
        fig_ts.add_scatter(
            x=anomalies_ts["date_hour"],
            y=anomalies_ts[y_col],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomalies"
        )
        fig_ts.update_layout(
            paper_bgcolor="#121212",
            plot_bgcolor="#1e1e1e",
            font_color="#e0e0e0",
            margin={"l": 40, "r": 40, "t": 60, "b": 40},
            xaxis_title="Date et Heure",
            yaxis_title="Moyenne de la métrique"
        )
        return html.Div([
            html.H3("Visualisation OLT", className="highlight"),
            dcc.Graph(figure=fig_ts, style={'width': '100%', 'height': '600px'})
        ])
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
