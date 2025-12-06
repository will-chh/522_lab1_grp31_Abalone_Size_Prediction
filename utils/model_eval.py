import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import click
import pandas as pd
import pickle

def evaluate_knn(
    knn,
    scaler,
    X_train,
    y_train,
    X_test,
    y_test,
    plot: bool = True,
):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("=== KNN Regression performance (k=5) ===")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test  RMSE : {test_rmse:.4f}")

    if plot:
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_test_pred, alpha=0.5)

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--",
            color="black",
            label="Perfect Prediction Line",
        )

        plt.xlabel("Actual number of rings (True age proxy)")
        plt.ylabel("Predicted number of rings")
        plt.title("KNN model performance: actual vs predicted rings")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return train_rmse, test_rmse

@click.command()
@click.option("--train_path", required=True, type=str, help="Path to train CSV")
@click.option("--test_path", required=True, type=str, help="Path to test CSV")
@click.option("--model_path", required=True, type=str, help="Path to saved KNN model (.pkl)")
@click.option("--scaler_path", required=True, type=str, help="Path to saved scaler (.pkl)")
@click.option("--no_plot", is_flag=True, help="Disable plotting")
def main(train_path, test_path, model_path, scaler_path, no_plot):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop("Rings", axis=1)
    y_train = train_df["Rings"]
    X_test = test_df.drop("Rings", axis=1)
    y_test = test_df["Rings"]

    # Load model and scaler
    with open(model_path, "rb") as f:
        knn = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Evaluate
    from __main__ import evaluate_knn  # import the function from this file
    evaluate_knn(knn, scaler, X_train, y_train, X_test, y_test, plot=not no_plot)

if __name__ == "__main__":
    main()