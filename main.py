import os
from apps.missing_data_app import retrieve_app_data, start_flask_app

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        retrieve_app_data()  # This will run only in the child process
    start_flask_app()
