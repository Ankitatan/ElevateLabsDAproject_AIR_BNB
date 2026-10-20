"""
Airbnb Listing Price Regression Model
Dataset: NYC Airbnb Open Data (2019)

Usage:
    pip install pandas numpy scikit-learn joblib
    python train_airbnb_model.py --data AB_NYC_2019.csv

The script:
- excludes non-positive prices
- prevents target leakage by never using `price` as an input feature
- imputes missing values and one-hot encodes categorical variables inside a pipeline
- uses a held-out test set and reports MAE, RMSE, and R²
- saves the trained model and a test-set predictions CSV
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


NUMERIC_CANDIDATES = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

CATEGORICAL_CANDIDATES = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
]


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    required = {"price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Convert target and numeric features safely.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only valid positive prices. Extreme positive prices are retained;
    # inspect them separately rather than silently deleting them.
    before = len(df)
    df = df.loc[df["price"].notna() & (df["price"] > 0)].copy()
    print(f"Rows loaded: {before:,}")
    print(f"Rows with valid positive price: {len(df):,}")
    print(f"Rows excluded for missing/non-positive price: {before - len(df):,}")

    if len(df) < 20:
        raise ValueError("Too few valid rows to train a reliable model.")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="AB_NYC_2019.csv",
        help="Path to the Airbnb CSV file (default: AB_NYC_2019.csv)",
    )
    parser.add_argument("--output-dir", default="airbnb_model_output")
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    numeric_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("No supported predictor columns were found in the dataset.")

    # Explicitly exclude price, IDs, host names, listing names, and free text.
    X = df[feature_columns].copy()
    y = df["price"].astype(float)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=args.trees,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Prices are strongly right-skewed; train on log(price) and convert
    # predictions back to the original price scale for reporting.
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\nTraining Random Forest regression model...")
    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n=== Held-out test-set metrics ===")
    print(f"MAE : ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²  : {r2:.4f}")
    print("\nNote: metrics are in the dataset's price units. Confirm currency and nightly-price meaning.")

    # Save model together with the exact feature list for reproducible inference.
    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "price",
        "training_note": "Positive-price NYC Airbnb listing model; log1p target transform; random 80/20 split.",
    }
    model_path = output_dir / "airbnb_price_model.joblib"
    joblib.dump(model_bundle, model_path)

    results = X_test.copy()
    results.insert(0, "actual_price", y_test.to_numpy())
    results["predicted_price"] = predictions
    results["absolute_error"] = np.abs(results["actual_price"] - results["predicted_price"])
    predictions_path = output_dir / "airbnb_test_predictions.csv"
    results.to_csv(predictions_path, index=False)

    metrics_path = output_dir / "model_metrics.txt"
    metrics_path.write_text(
        "Airbnb Price Regression — held-out test set\n"
        f"Rows used: {len(df):,}\n"
        f"Training rows: {len(X_train):,}\n"
        f"Test rows: {len(X_test):,}\n"
        f"Features: {', '.join(feature_columns)}\n"
        f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.6f}\n",
        encoding="utf-8",
    )

    print("\nSaved files:")
    print(f"- Model:      {model_path}")
    print(f"- Predictions:{predictions_path}")
    print(f"- Metrics:    {metrics_path}")
    print("\nModel limitation: this predicts listing prices from available listing attributes.")
    print("It does not establish causal effects or learn true seasonal demand from this snapshot.")


if __name__ == "__main__":
    main()
