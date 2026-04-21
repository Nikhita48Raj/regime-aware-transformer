import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_etth1(csv_path: str, target_col: str = "OT"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    feature_cols = [col for col in df.columns if col != "date"]
    target_idx = feature_cols.index(target_col)

    data = df[feature_cols].values

    return df, data, feature_cols, target_idx


def train_val_test_split(data, train_ratio=0.7, val_ratio=0.1):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


def scale_splits(train, val, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    return train_scaled, val_scaled, test_scaled, scaler