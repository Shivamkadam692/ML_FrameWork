import pandas as pd

def load_csv(filepath, **kwargs):
    """Load dataset from a CSV file using pandas."""
    return pd.read_csv(filepath, **kwargs)

def handle_missing_values(df, strategy='mean'):
    """Fill missing values in a pandas DataFrame."""
    if strategy == 'mean':
        # Select only numeric columns before calculating mean
        numeric_df = df.select_dtypes(include='number')
        df[numeric_df.columns] = df[numeric_df.columns].fillna(numeric_df.mean())
        return df
    elif strategy == 'median':
        numeric_df = df.select_dtypes(include='number')
        df[numeric_df.columns] = df[numeric_df.columns].fillna(numeric_df.median())
        return df
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
