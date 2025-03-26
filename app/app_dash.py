import dash
from dash import dcc, html, Input, Output, State, ctx
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
external_stylesheets = [{
    "href": "https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;700&display=swap",
    "rel": "stylesheet",
}]

# Personnalisation de l'index – ici on ne surcharge que l’input du DatePickerSingle
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        body {
            margin: 0;
            background-color: #1e1e2f;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            bottom: 0;
            width: 220px;
            background-color: #2b2b3d;
            padding: 20px;
            overflow-y: auto;
        }
        .sidebar h2 {
            color: #00bcd4;
            font-weight: bold;
        }
        .sidebar a {
            display: block;
            color: #f0f0f0;
            text-decoration: none;
            margin: 10px 0;
            font-weight: bold;
        }
        .sidebar a:hover {
            color: #00bcd4;
        }
        .content {
            margin-left: 240px;
            padding: 20px;
        }
        .card {
            background-color: #2c2f48;
            padding: 15px;
            border-radius: 10px;
            margin: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.4);
        }
        .highlight {
            color: #00bcd4;
            font-weight: bold;
        }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>📡 Dashboard</h2>
            <a href="/dashboard" id="tab-dashboard">🚨 Alertes</a>
            <a href="/journal" id="tab-journal">📋 Journal</a>
            <a href="/map" id="tab-map">🗺 Carte</a>
            <a href="/visual" id="tab-visual">📊 Visualisation</a>
        </div>
        <div class="content">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout de l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
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
            style={'color': '#00bcd4', 'backgroundColor': '#1e1e2f'}
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
        # Bloc pour l'onglet Visualisation (affiché uniquement sur la page /visual)
        html.Div([
            html.Label("OLT", className="highlight"),
            dcc.Dropdown(
                id='olt-picker',
                options=[],
                placeholder="Sélectionnez un OLT",
                style={'color': '#00bcd4', 'backgroundColor': '#1e1e2f'}
            )
        ], id="olt-visual-controls", style={"display": "none", "marginTop": "10px"})
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'marginLeft': '20px'}),
    html.Div(id="page-content", style={'width': '70%', 'display': 'inline-block', 'padding': '10px'})
])

# Callback pour mettre à jour le sélecteur OLT (uniquement sur la page Visualisation)
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

# Callback pour afficher le contenu en fonction de l'URL (page active)
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('hours-back', 'value'),
    Input('anomaly-type', 'value'),
    Input('olt-picker', 'value')
)
def display_page(pathname, date_str, hour, hours_back, anomaly_type, selected_olt):
    if pathname not in ['/dashboard', '/journal', '/map', '/visual']:
        pathname = '/dashboard'

    selected_datetime = datetime.combine(pd.to_datetime(date_str).date(), time(int(hour)))
    start_datetime = selected_datetime - timedelta(hours=int(hours_back))
    selected_anomaly_col = anomaly_col_map[anomaly_type]

    if pathname == '/dashboard':
        historical_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] <= selected_datetime)].copy()
        historical_df["anomalie"] = historical_df[selected_anomaly_col]
        # Affichage de 4 attributs
        attrs = ["code_departement", "type_olt_model", "type_boucle", "boucle"]

        # Section 1 : Graphiques en camembert (pour "code_departement" et "boucle", ne garder que les 3 principales valeurs)
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
                legend=dict(font_color='#f0f0f0'),
                font=dict(size=10)
            )
            pie_charts.append(html.Div(
                dcc.Graph(figure=pie_fig, config={'displayModeBar': False}),
                style={'width': '300px', 'margin': '10px'}
            ))

        # Section 2 : Détails des volumes
        volume_details = []
        for attr in attrs:
            df_attr = historical_df[historical_df["anomalie"]]
            total_volume = df_attr.shape[0]
            limit = 3 if attr in ["code_departement", "boucle"] else 5
            counts = df_attr.groupby(attr).size().reset_index(name="nb_alertes")
            top_items = counts.sort_values("nb_alertes", ascending=False).head(limit)
            top_list = html.Ul([
                html.Li(f"{row[attr]} : {row['nb_alertes']} alertes") for _, row in top_items.iterrows()
            ])
            volume_details.append(html.Div([
                html.H4(attr.replace('_', ' ').title(), className="highlight"),
                html.P(f"Volume total d'anomalies: {total_volume}", style={'fontSize': '12px'}),
                top_list
            ], style={
                'backgroundColor': '#2c2f48',
                'padding': '15px',
                'borderRadius': '10px',
                'margin': '10px',
                'width': '300px'
            }))

        return html.Div([
            html.H3("🚨 Détails des volumes d'anomalies", className="highlight"),
            html.Div(volume_details, style={"display": "flex", "flexWrap": "wrap"}),
            html.H3("🚨 Répartition des anomalies", className="highlight"),
            html.Div(pie_charts, style={"display": "flex", "flexWrap": "wrap"})
        ])

    elif pathname == '/journal':
        journal_df = df[(df["date_hour"] >= start_datetime) & (df["date_hour"] < selected_datetime)]
        journal_df = journal_df[journal_df[selected_anomaly_col]]
        journal_df = journal_df.sort_values("date_hour", ascending=False)
        friendly_columns = [{"name": col.replace("is_jump", "anomalie").replace("_", " ").title(), "id": col} for col in journal_df.columns]
        buffer = io.StringIO()
        journal_df.to_csv(buffer, index=False)
        csv_data = base64.b64encode(buffer.getvalue().encode()).decode()
        download_link = html.A(
            "⬇️ Exporter en CSV",
            href=f"data:text/csv;base64,{csv_data}",
            download="journal_anomalies.csv",
            style={"marginTop": "10px", "display": "inline-block", "color": "#00bcd4", "fontWeight": "bold"}
        )
        return html.Div([
            html.H3("📋 Journal des anomalies", className="highlight"),
            dash.dash_table.DataTable(
                data=journal_df.to_dict("records"),
                columns=friendly_columns,
                style_table={'overflowX': 'auto'},
                page_size=20,
                filter_action="native",
                sort_action="native",
                style_header={'backgroundColor': '#2c2f48', 'color': '#f0f0f0', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#1e1e2f', 'color': '#f0f0f0'}
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
            html.H3("🗺 Carte des anomalies", className="highlight"),
            dcc.Graph(figure=fig_map)
        ])

    elif pathname == '/visual':
        if not selected_olt:
            return html.Div([html.H4("📊 Visualisation", className="highlight"), html.P("Veuillez sélectionner un OLT")])
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
            paper_bgcolor="#1e1e2f",
            plot_bgcolor="#2c2f48",
            font_color="#f0f0f0",
            margin={"l": 40, "r": 40, "t": 60, "b": 40},
            xaxis_title="Date et Heure",
            yaxis_title="Moyenne de la métrique"
        )
        return html.Div([
            html.H3("📊 Visualisation OLT", className="highlight"),
            dcc.Graph(figure=fig_ts)
        ])

    return html.Div(["Page non reconnue"])

if __name__ == '__main__':
    app.run(debug=True)
