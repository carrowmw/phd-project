import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Scales all values in the DataFrame between 0 and 1 using MinMaxScaler.

    Args:
        df (pd.DataFrame): DataFrame with the data to be scaled.

    Returns:
        pd.DataFrame: The scaled version of the input data as a DataFrame.
    """
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)  # Apply scaler to df directly
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    return scaled_df


def resample_frequency(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Resamples a DataFrame using a given frequency, calculating mean and standard deviation for
    each resampled period. Used primarily to downsample time-series data, summarizing with mean
    and standard deviation.

    Args:
        df (pd.DataFrame): Input DataFrame expected to have a DateTimeIndex.
        frequency (str, optional): Frequency for resampling the data. Can be any valid frequency
                                   alias in pandas. Default is None.

    Returns:
        pd.DataFrame: DataFrame resampled to the provided frequency with 'mean' and 'std'.
    """
    frequency = kwargs.get("frequency")
    # Check if frequency is provided; if not, return the original DataFrame
    if frequency is None:
        print("No frequency provided for resampling. Returning original DataFrame.")
        return df

    # Proceed with resampling if frequency is provided
    resampled_df = df.resample(frequency).agg({"value": ["mean", "std"]})
    resampled_df.columns = [
        "mean",
        "std",
    ]  # Flatten the MultiIndex columns after aggregation
    return resampled_df


def add_term_dates_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds term-related features 'newcastle_term' and 'northumbria_term' to the input DataFrame based on date ranges.

    Args:
        df (pandas.DataFrame): The DataFrame to which the new features will be added. It should have a datetime index.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns 'newcastle_term' and 'northumbria_term'.
                          These columns represent binary indicators for the terms of each university.

    Note:
        The function assumes that the input DataFrame `df` has a datetime index. Additionally, the term start and end
        dates for Newcastle and Northumbria universities are hardcoded in the function, so any changes to the term
        dates should be made directly in the function code.
    """
    print("Adding term dates to dataframes")
    # Define the date range for the series
    start = min(df.index.min(), df.index.min())
    end = max(df.index.max(), df.index.max())
    date_range = pd.date_range(start=start, end=end, freq="15min")

    # Define the start and end dates for each term
    newcastle_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-03-25"),
        ("2022-04-25", "2022-06-17"),
    ]
    newcastle_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-24", "2023-06-16"),
    ]
    northumbria_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-04-01"),
        ("2022-04-25", "2022-05-27"),
    ]
    northumbria_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-17", "2023-06-02"),
    ]

    # Create binary series for each term of each university
    newcastle_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2122
    ]
    newcastle_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2223
    ]
    northumbria_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2122
    ]
    northumbria_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2223
    ]

    # Combine the binary series for each university into a single series
    newcastle = pd.concat(newcastle_2122 + newcastle_2223, axis=1).max(axis=1)
    northumbria = pd.concat(northumbria_2122 + northumbria_2223, axis=1).max(axis=1)

    # Add the new features to the input DataFrame df
    df["newcastle_term"] = newcastle.astype(bool)
    df["northumbria_term"] = northumbria.astype(bool)

    return df


def create_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates frequency features for time series data and adds these features to the input DataFrame.
    The function generates sine and cosine features based on daily, half-day, quarter-yearly,
    and yearly periods to capture potential cyclical patterns.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with a DatetimeIndex containing the timestamps for which frequency features are to be created.

    Returns
    -------
    df : pandas DataFrame
        The input DataFrame, with the following new columns:
        - 'sin_day': Sine of the time of day, assuming a period of 24 hours.
        - 'cos_day': Cosine of the time of day, assuming a period of 24 hours.
        - 'sin_half_day': Sine of the time of day, assuming a period of 12 hours.
        - 'cos_half_day': Cosine of the time of day, assuming a period of 12 hours.
        - 'sin_quarter': Sine of the day of the year, assuming a period of about 91.25 days.
        - 'cos_quarter': Cosine of the day of the year, assuming a period of about 91.25 days.
        - 'sin_year': Sine of the day of the year, assuming a period of 365 days.
        - 'cos_year': Cosine of the day of the year, assuming a period of 365 days.
    """
    # Make a copy of the input DataFrame to avoid modifying it
    df = df.copy()
    dt_index = df.index

    df["sin_half_day"] = np.sin(2 * np.pi * dt_index.hour / 12)
    df["cos_half_day"] = np.cos(2 * np.pi * dt_index.hour / 12)
    df["sin_day"] = np.sin(2 * np.pi * dt_index.hour / 24)
    df["cos_day"] = np.cos(2 * np.pi * dt_index.hour / 24)
    df["sin_week"] = np.sin(2 * np.pi * dt_index.isocalendar().week / 52)
    df["cos_week"] = np.cos(2 * np.pi * dt_index.isocalendar().week / 52)
    df["sin_quarter"] = np.sin(2 * np.pi * dt_index.dayofyear / 91.25)
    df["cos_quarter"] = np.cos(2 * np.pi * dt_index.dayofyear / 91.25)
    df["sin_year"] = np.sin(2 * np.pi * dt_index.dayofyear / 365)
    df["cos_year"] = np.cos(2 * np.pi * dt_index.dayofyear / 365)
    return df


def create_anomaly_column(timeseries_array: np.ndarray) -> np.ndarray:
    """
    Add a new boolean column to a given time series array indicating anomalies.

    The new column will have a True value if the corresponding value in the first
    column of the original array is greater than 0.9, otherwise, it will have a False value.

    Parameters:
    - timeseries_array (np.ndarray): A 2D numpy array where the first column represents
      the time series values.

    Returns:
    - np.ndarray: The original numpy array concatenated with the new boolean column indicating
    anomalies.

    Example:
    >>> arr = np.array([[0.8], [0.92], [0.85], [0.95]])
    >>> create_anomaly_column(arr)
    array([[0.8 , 0.  ],
           [0.92, 1.  ],
           [0.85, 0.  ],
           [0.95, 1.  ]])
    """
    # Create a new boolean column where True indicates that the value in the first column is greater than 0.9
    anomaly = timeseries_array[:, 0] > 0.9
    # Reshape the new column to be two-dimensional so it can be concatenated with the original array
    anomaly = anomaly.reshape(-1, 1)
    # Concatenate the new column to your array along axis 1
    timeseries_array = np.concatenate((timeseries_array, anomaly), axis=1)
    return timeseries_array


def extract_time_features(df):
    """
    Extracts time-related features from a 'Timestamp' column and then removes it.

    Args:
        df (pd.DataFrame): DataFrame with a 'Timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with 'Timestamp' split into separate features and the original 'Timestamp' column removed.
    """
    # Ensure 'Timestamp' is in datetime format
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Extract features
    df["Year"] = df["Timestamp"].dt.year.astype(float)
    df["Month"] = df["Timestamp"].dt.month.astype(float)
    df["Day"] = df["Timestamp"].dt.day.astype(float)
    df["Hour"] = df["Timestamp"].dt.hour.astype(float)
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek.astype(float)
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear.astype(float)

    # Remove 'Timestamp' column
    df.drop(columns=["Timestamp"], inplace=True)

    return df
