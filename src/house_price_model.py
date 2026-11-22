
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"

FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "House_Age",
]

TARGET = "SalePrice"


def load_data(file_path=DATA_PATH):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def validate_data(data):
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    return data


def train_model(data):
    data = validate_data(data)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    }


def predict_house_price(model, area, bedrooms, bathrooms, house_age):
    house_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, house_age]],
        columns=FEATURES,
    )

    return float(model.predict(house_data)[0])


def main():
    data = load_data()
    model, X_test, y_test = train_model(data)
    metrics = evaluate_model(model, X_test, y_test)

    print("House Price Prediction")
    print("-" * 30)
    print(f"Dataset records: {len(data)}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:,.4f}")

    print("Model training completed successfully.")


if __name__ == "__main__":
    main()
