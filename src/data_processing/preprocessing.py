import pandas as pd
from datetime import datetime, timedelta
from src.utils.general_utils import load_config


def remove_directionality_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Removes directionality in data by aggregating values with the same timestamp, effectively
    summing the 'value' for each group. Useful for datasets where 'value' depends on a directional
    parameter and considering the total amount regardless of the direction is desired.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Timestamp' for datetime and 'value'.
        **kwargs: Arbitrary keyword arguments. 'additional_features' expected as a list of features to include in the aggregation.

    Returns:
        pd.DataFrame: DataFrame with directionality removed, indexed by 'Timestamp' with summed 'value'.
    """
    agg_dict = {"Value": "sum"}
    features = kwargs.get("features_to_include_on_aggregation", [])
    if features:
        for feature in features:
            agg_dict[feature] = "first"
    df = df.groupby("Timestamp").agg(agg_dict).reset_index()
    return df


def compute_max_daily_records(df: pd.DataFrame) -> int:

    # Calculate differences between consecutive timestamps
    df["Time_Difference"] = df["Timestamp"].diff()

    # Convert time differences to a total number of minutes for easier analysis
    df["Interval_Minutes"] = df["Time_Difference"].dt.total_seconds() / 60

    # Handle case where 'Interval_Minutes' might have NaN values
    min_interval = df["Interval_Minutes"].min()
    if pd.isna(min_interval):
        print(
            "Warning: No valid time intervals found. Defaulting max_daily_records to NaN."
        )
        return float("nan")

    max_daily_records = 24 * 60 / min_interval
    return max_daily_records


def remove_incomplete_days(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Select data from a sensor dataframe based on completeness threshold.

    Args:
        df (pandas.DataFrame): The dataframe.
        completeness_threshold (float): The completeness threshold (ranging from 0 to 1).

    Returns:
        pandas.DataFrame: The selected dataframe based on the completeness threshold.
    """
    preprocessing_config = load_config("configs/preprocessing_config.json")
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]

    # Assuming compute_max_daily_records is defined elsewhere and compatible with this approach
    threshold = completeness_threshold * compute_max_daily_records(df)

    # Extract date component from the Timestamp column
    if "Timestamp" in df.columns:
        # If Timestamp is a column
        df["Date"] = df["Timestamp"].dt.date
    else:
        # If the DataFrame is indexed by Timestamp
        df["Date"] = df.index.date

    # Group by the extracted 'Date' column and count the number of entries for each date
    date_counts = df.groupby("Date").size()

    # Find the dates that have at least the threshold number of entries
    valid_dates = date_counts[date_counts >= threshold].index

    # Select only the rows that have a date in valid_dates
    complete_days_df = df[df["Date"].isin(valid_dates)].drop(columns=["Date"])

    print(f"Initial number of records: {len(df)}")
    print(
        f"Number of records in days @ {completeness_threshold * 100:.0f}% completeness: {len(complete_days_df)}"
    )
    print(
        f"Proportion of records removed: {(1 - len(complete_days_df)/len(df))*100:.2f}%"
    )

    return complete_days_df


def check_completeness(df: pd.DataFrame):
    """
    Check if data completeness for each day in a sequence meets the threshold, based on configuration settings.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data, expected to have a datetime index or a datetime column.

    Returns:
        bool: True if the completeness meets the threshold for each day in the sequence, False otherwise.
    """
    # Load configurations
    api_config = load_config("configs/api_config.json")
    preprocessing_config = load_config("configs/preprocessing_config.json")

    # Determine start and end dates based on today and last_n_days from api_config
    end_date = datetime.today()
    start_date = end_date - timedelta(
        days=api_config["api"]["endpoints"]["raw_sensor_data"]["params"]["last_n_days"]
    )

    # Get completeness threshold from preprocessing_config
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]

    # Ensure DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    expected_records_per_day = compute_max_daily_records(df)

    # Filter DataFrame for the specified date range and check completeness
    for day in pd.date_range(start=start_date, end=end_date):
        daily_records = df[day.strftime("%Y-%m-%d")]
        if len(daily_records) < completeness_threshold * expected_records_per_day:
            return False

    return True


def find_longest_consecutive_sequence(df: pd.DataFrame, **kwargs):
    """
    Finds the longest sequence of consecutive days in the DataFrame `df` where
    the data completeness meets or exceeds a specified threshold, with an option
    to limit the search to a maximum sequence length.

    Args:
        df (pd.DataFrame): The DataFrame containing time-series data indexed by datetime.
        completeness_threshold (float): The data completeness threshold to apply.
        max_length_limit (int): Optional. The maximum length of the sequence to search for before stopping.
                                If None, the search will continue until the end of the dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the longest sequence of days meeting the completeness criteria.
                      If no sequence meets the criteria, returns an empty DataFrame.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index("Timestamp", inplace=True, drop=False)
    preprocessing_config = load_config("configs/preprocessing_config.json")
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]

    max_length_limit = kwargs.get("max_length_limit")
    max_daily_records = compute_max_daily_records(df)
    if pd.isna(max_daily_records):
        print(
            "Warning: max_daily_records is NaN. Unable to compute longest consecutive sequence."
        )
        return pd.DataFrame()
    longest_sequence = pd.DataFrame()
    current_sequence_start = None
    current_length = 0
    max_length = 0

    for day in pd.date_range(df.index.min(), df.index.max()):
        daily_records = df[df.index.date == day.date()]
        if len(daily_records) >= completeness_threshold * max_daily_records:
            if current_sequence_start is None:
                current_sequence_start = day
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                longest_sequence = df.loc[current_sequence_start:day]
        else:
            current_sequence_start = None
            current_length = 0
            # Early termination if the current sequence length meets the max_length_limit
            if max_length_limit is not None and max_length >= max_length_limit:
                break
    print(f"Longest consecutive sequence is {max_length}")
    return longest_sequence


def remove_specified_fields(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Removes specified columns from a DataFrame based on kwargs input.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be removed.
    - kwargs (dict): Keyword arguments specifying which columns to remove. Expected to find a key 'columns_to_drop' that contains a list of column names to be removed.

    Returns:
    - pd.DataFrame: A DataFrame with the specified columns removed.
    """
    columns_to_drop = kwargs.get("columns_to_drop", [])
    if not columns_to_drop:
        print("No columns specified for removal.")
        return df

    # Ensure all specified columns exist in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop specified columns
    df = df.drop(columns=columns_to_drop)

    print(f"Removed columns: {columns_to_drop}")
    return df
