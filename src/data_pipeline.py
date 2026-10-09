import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(file_path):
    df = load_data(file_path)

    features = df[["attendance", "assignment_score", "quiz_score", "participation", "study_hours"]]
    target = df["needs_support"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target
    )

    return X_train, X_test, y_train, y_test