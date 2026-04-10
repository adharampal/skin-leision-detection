import os
import pandas as pd
import kagglehub
from sklearn.model_selection import GroupShuffleSplit


def load_data():
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    csv_path = os.path.join(path, "HAM10000_metadata.csv")
    return pd.read_csv(csv_path)


def preprocess(data):
    cleaned_df = data.copy()
    cleaned_df = cleaned_df.drop(columns=["image_id"])

    cleaned_df["age"] = cleaned_df["age"].fillna(cleaned_df["age"].median())

    malignant = ["mel", "bcc", "akiec"]
    cleaned_df["target"] = cleaned_df["dx"].isin(malignant).astype(int)

    X = cleaned_df.drop(columns=["dx", "target", "lesion_id", "dx_type"])
    y = cleaned_df["target"]

    X = pd.get_dummies(X, columns=["sex", "localization"], drop_first=True)

    return X, y, cleaned_df["lesion_id"]


def split_data(X, y, groups, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
    )
