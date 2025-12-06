import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    FeatureLabelCorrelation,
    FeatureFeatureCorrelation,
    LabelDrift,
)
import click

def preprocess_and_split(
    abalone: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1. One-hot encode Sex
    abalone_converted = pd.get_dummies(abalone, columns=["Sex"], drop_first=True)

    X = abalone_converted.drop("Rings", axis=1)
    y = abalone_converted["Rings"]

    # 3. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    abalone_train = X_train.copy()
    abalone_train["Rings"] = y_train

    train_ds = Dataset(
        abalone_train.drop(columns=["Sex_I", "Sex_M"]),
        label="Rings",
        cat_features=[],
    )

    # 1) Feature–Label correlation
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(
        0.9
    )
    res_feat_lab = check_feat_lab_corr.run(dataset=train_ds)

    # 2) Feature–Feature correlation
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.99,
        n_pairs=0,
    )
    res_feat_feat = check_feat_feat_corr.run(dataset=train_ds)

    # 3) Label drift
    abalone_test = X_test.copy()
    abalone_test["Rings"] = y_test

    test_ds = Dataset(
        abalone_test.drop(columns=["Sex_I", "Sex_M"]),
        label="Rings",
        cat_features=[],
    )

    check_label_drift = LabelDrift().add_condition_drift_score_less_than(0.2)
    res_label_drift = check_label_drift.run(
        train_dataset=train_ds, test_dataset=test_ds
    )

    if not res_feat_lab.passed_conditions():
        raise ValueError("Feature–Label correlation exceeds the acceptable threshold.")
    if not res_feat_feat.passed_conditions():
        raise ValueError("Feature–Feature correlation exceeds the acceptable threshold.")
    if not res_label_drift.passed_conditions():
        raise ValueError("Target variable distribution drift detected.")

    return X_train, X_test, y_train, y_test


# output for all vars
@click.command()
@click.option("--input_path", required=True, type=str, help="Path to validated CSV")
@click.option("--train_output", required=True, type=str, help="Path to save train CSV")
@click.option("--test_output", required=True, type=str, help="Path to save test CSV")
@click.option("--test_size", default=0.2, type=float, help="Test set proportion")
@click.option("--random_state", default=42, type=int, help="Random seed")
def main(input_path, train_output, test_output, test_size, random_state):
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = preprocess_and_split(df, test_size, random_state)

    train_df = X_train.copy()
    train_df["Rings"] = y_train
    test_df = X_test.copy()
    test_df["Rings"] = y_test

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

if __name__ == "__main__":
    main()
