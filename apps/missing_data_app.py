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

from src.data_processing.execute_requests import (
    execute_sensors_request,
    execute_raw_sensor_data_request,
    get_daily_counts_dataframes,
    print_sensor_request_metrics,
)

from src.utils.app_utils import (
    create_and_load_file_path,
    save_data_to_file,
    load_data_from_file,
)


def retrieve_app_data() -> list:
    """
    Retrieves and formats application data for visualization in a missing data app.

    It fetches data based on the configurations specified in 'api_config.json', including
    the bounding box coordinates and the number of days for which data is required. If
    data for the given period exists locally, it is read from the storage; otherwise, it is
    requested from the sensor API and processed.

    Returns:
        list: A list of dataframes containing the daily counts of records for each sensor.
    """
    app_data_directory = "daily_record_counts"
    api_config_file_path = "configs/api_config.json"
    file_path = create_and_load_file_path(api_config_file_path, app_data_directory)
    app_data = load_data_from_file(file_path)
    if app_data is not None:
        return app_data

    sensors_df = execute_sensors_request()
    series_of_sensor_names = sensors_df["Sensor Name"]
    list_of_raw_data_dfs = execute_raw_sensor_data_request(sensors_df)
    print_sensor_request_metrics(list_of_raw_data_dfs, series_of_sensor_names)
    app_data = get_daily_counts_dataframes(list_of_raw_data_dfs)
    save_data_to_file(file_path, app_data)
    return app_data


def create_dashboard(dataframes, last_n_days):
    """
    Initializes and configures a Dash application to visualize sensor data.

    The dashboard visualizes the daily counts of records from various sensors over the
    last specified number of days using bar charts. Each sensor's data is represented in
    its own graph, displaying the percentage of data completeness.

    Args:
        dataframes (list): A list of tuples, where each tuple contains the sensor name and
                           its corresponding dataframe with records counts.
        last_n_days (int): The number of days for which data is visualized.

    Returns:
        dash.Dash: A Dash application configured with the necessary layout and callbacks
                   for data visualization.
    """
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
    """
    Starts the Flask application with the Dash dashboard for visualizing sensor data.

    Reads the configuration for the last number of days to display from 'api_config.json',
    retrieves application data using `retrieve_app_data`, initializes the dashboard with
    `create_dashboard`, and starts the Flask server to host the Dash app. The web app is
    automatically opened in a browser upon launch.
    """
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


# Run missing dashboard app
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        retrieve_app_data()  # This will run only in the child process
    start_flask_app()
