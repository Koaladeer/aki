import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def convert_change_percent(value):
    if isinstance(value, str):
        value = value.replace('%', '')  # Remove percentage sign
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000  # Convert millions to numeric
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000  # Convert thousands to numeric
        else:
            return float(value)  # Default conversion
    return value

# Function to encode a date into cyclical features
def encode_date_column(df, column):
    """
    Encode a date column into cyclical features (sine and cosine) and return as a DataFrame.

    Parameters:
    - df: Original DataFrame
    - column: Column name to encode

    Returns:
    - A DataFrame with encoded features.
    """
    date_series = df[column]

    # Extract components
    years = date_series.dt.year
    months = date_series.dt.month
    days = date_series.dt.day

    # Cyclical encoding
    months_sin = np.sin(2 * np.pi * months / 12)
    months_cos = np.cos(2 * np.pi * months / 12)
    days_sin = np.sin(2 * np.pi * days / 31)
    days_cos = np.cos(2 * np.pi * days / 31)

    # Create encoded DataFrame
    encoded_df = pd.DataFrame({
        f'{column}_year': years,
        f'{column}_month_sin': months_sin,
        f'{column}_month_cos': months_cos,
        f'{column}_day_sin': days_sin,
        f'{column}_day_cos': days_cos
    })

    return encoded_df


def numeric_encode_column(series):
    """
    Perform numeric encoding for a datetime column

    Args:
        series (pd.Series): Input datetime series

    Returns:
        pd.DataFrame: DataFrame with numeric encoded features
    """
    # Ensure the series is datetime
    series = pd.to_datetime(series, errors='coerce')

    # Method 1: Unix Timestamp
    timestamp_encoding = series.astype(int) // 10 ** 9

    # Method 2: Days since reference date
    reference_date = series.min()
    days_encoding = (series - reference_date).dt.days

    # Method 3: Separate features
    year_encoding = series.dt.year
    month_encoding = series.dt.month
    day_encoding = series.dt.day

    # Create scaler
    scaler = MinMaxScaler()

    # Combine and normalize
    multi_feature_encoding = np.column_stack([
        scaler.fit_transform(timestamp_encoding.values.reshape(-1, 1)),
        scaler.fit_transform(days_encoding.values.reshape(-1, 1)),
        scaler.fit_transform(year_encoding.values.reshape(-1, 1)),
        scaler.fit_transform(month_encoding.values.reshape(-1, 1)),
        scaler.fit_transform(day_encoding.values.reshape(-1, 1))
    ])

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        multi_feature_encoding,
        columns=[
            f'{series.name}_timestamp',
            f'{series.name}_days_since_ref',
            f'{series.name}_year',
            f'{series.name}_month',
            f'{series.name}_day'
        ],
        index=series.index
    )

    return encoded_df
