import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import click
import pickle

def fit_knn_regressor(
    X_train,
    y_train,
    n_neighbors: int = 5,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    knn.fit(X_train_scaled, y_train)

    return knn, scaler

@click.command()
@click.option("--train_path", required=True, type=str, help="Path to train CSV")
@click.option("--model_output", required=True, type=str, help="Path to save trained KNN model (.pkl)")
@click.option("--scaler_output", required=True, type=str, help="Path to save fitted scaler (.pkl)")
@click.option("--n_neighbors", default=5, type=int, help="Number of neighbors for KNN")
def main(train_path, model_output, scaler_output, n_neighbors):
    # Load training data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop("Rings", axis=1)
    y_train = train_df["Rings"]

    # Fit KNN
    # from __main__ import fit_knn_regressor  # imports the function from this file
    knn, scaler = fit_knn_regressor(X_train, y_train, n_neighbors)

    # Save model and scaler
    with open(model_output, "wb") as f:
        pickle.dump(knn, f)
    with open(scaler_output, "wb") as f:
        pickle.dump(scaler, f)

    print(f"KNN model saved to {model_output}")
    print(f"Scaler saved to {scaler_output}")
    print("Model fitting completed successfully.")

if __name__ == "__main__":
    main()