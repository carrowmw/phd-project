import os
from datetime import date
import json
import webbrowser
from datetime import datetime, timedelta
from threading import Timer
import pandas as pd
import pickle
import dash
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

from data_processing.execute_requests import (
    execute_sensors_request,
    execute_raw_sensor_data_request,
    get_daily_counts_dataframes,
    print_sensor_request_metrics,
)

from src.utils.polygon_utils import create_wkb_polygon


def retrieve_app_data() -> list:
    """
    Retrieves and formats the data for use in the missing data app
    """
    today = date.today()
    with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
        api_config = json.load(config_file)
    last_n_days = api_config["api"]["endpoints"]["raw_sensor_data"]["params"][
        "last_n_days"
    ]
    coords = api_config["api"]["coords"]
    bbox = create_wkb_polygon(coords[0], coords[1], coords[2], coords[3])
    file_path = f"data/processed/daily_records_counts/{today}_Last_{last_n_days}_Days_{bbox}.pkl"
    if os.path.exists(file_path):
        print("\nReading in app data from local storage\n")
        with open(file_path, "rb") as f:
            app_data = pickle.load(f)
        return app_data

    sensors_df = execute_sensors_request()
    series_of_sensor_names = sensors_df["Sensor Name"]
    list_of_raw_data_dfs = execute_raw_sensor_data_request(sensors_df)
    print_sensor_request_metrics(list_of_raw_data_dfs, series_of_sensor_names)
    app_data = get_daily_counts_dataframes(list_of_raw_data_dfs)
    # Extract the directory path
    dir_path = os.path.dirname(file_path)

    # Create the directory (and any intermediate directories) if they don't exist
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(app_data, f)
    return app_data


def create_dashboard(dataframes, last_n_days):
    # Create a Dash web app
    app = dash.Dash(__name__)

    # Layout of the web app
    app.layout = html.Div(
        children=[
            # Container for the list of graphs
            html.Div(
                id="graphs-container",
                style={
                    "maxWidth": "2400",
                    "margin": "auto",
                    "textAlign": "center",
                },  # Center the content
            ),
        ]
    )

    # Callback to update the graph based on the selected DataFrame
    @app.callback(
        Output("graphs-container", "children"),
        [Input("graphs-container", "id")],  # Use any input to trigger the callback
    )
    def update_graph(_):
        try:
            # Create a list of html.Div elements for each graph
            graph_divs = []
            for i, _ in enumerate(dataframes):
                df = dataframes[i][1]
                sensor_name = dataframes[i][0]

                # Set x-axis and y-axis limits
                today = datetime.now()
                x_max = today.strftime("%Y-%m-%d")
                x_min = (today - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
                y_max = 96 * 3
                y_min = 0

                graph_divs.append(
                    html.Div(
                        children=[
                            dcc.Graph(
                                id=f"graph-{i}",
                                figure=px.bar(
                                    df,
                                    x="Timestamp",
                                    y="Count",
                                    title=(
                                        f"{sensor_name}   |   "
                                        f"{len(df)/(last_n_days*96)}% Complete"
                                    ),
                                    labels={
                                        "Count": "Total Records",
                                        "Timestamp": "Date",
                                    },
                                ).update_layout(
                                    title={
                                        "text": sensor_name,
                                        "font": {"size": 24},
                                        "x": 0.5,  # Center the title horizontally
                                    },
                                    xaxis=dict(
                                        range=[x_min, x_max],
                                        title="Date",
                                    ),
                                    yaxis=dict(
                                        range=[y_min, y_max],
                                        title="Count",
                                    ),
                                ),
                                style={"width": "100%"},  # Make each graph 100% width
                            ),
                        ],
                        id=f"section-{i}",
                        style={
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "height": "500px",
                            "width": "100%",  # Make each graph 100% width
                        },
                    )
                )

            return graph_divs
        except Exception as e:
            print(f"Error updating graphs-container.children: {str(e)}")
            return []

    print("\n \n    Web-app successfully initialized...")
    return app


def start_flask_app():
    """Function to start Flask application."""

    # Last n days needed for graph width
    with open("configs/api_config.json", "r", encoding="utf-8") as config_file:
        api_config = json.load(config_file)
    last_n_days = api_config["api"]["endpoints"]["raw_sensor_data"]["params"][
        "last_n_days"
    ]
    app_data = retrieve_app_data()
    dashboard_app = create_dashboard(
        app_data, last_n_days
    )  # Assuming create_dashboard is appropriately defined

    HOST = "127.0.0.1"
    PORT = 8050
    url = f"http://{HOST}:{PORT}"

    def open_browser():
        webbrowser.open_new(url)

    Timer(1, open_browser).start()
    dashboard_app.run_server(debug=True, host=HOST, port=PORT)
