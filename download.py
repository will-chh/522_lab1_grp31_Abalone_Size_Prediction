from pandera import Column, Check, DataFrameSchema
import pandas as pd

import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation, MultivariateDrift, LabelDrift
from deepchecks.tabular import Dataset

# Allowed categories for Sex
SEX_CATEGORIES = ["M", "F", "I"]

abalone_schema = DataFrameSchema(
    {
        "Sex": Column(
            str,
            Check.isin(SEX_CATEGORIES),
            nullable=False
        ),
        "Length": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Diameter": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Height": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Whole_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Shucked_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Viscera_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Shell_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Rings": Column(
            int,
            Check.between(1, 30),
            nullable=False
        )
    },
    checks=[
        Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
        Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
        Check(lambda df: (df.isna().mean() <= 0.05).all(),
              error="Missingness exceeds 5% threshold.")
    ]
)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

column_names = [
    "Sex", "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

#This ensures our data is freshly loaded everytime :) 
#Creating a function here for the loading process: 

def load_and_validate_abalone() -> pd.DataFrame:
    #1) Loading data 
    abalone_raw = pd.read_csv(url, header=None, names=column_names)

    #2) validation with pandera
    abalone_validated = abalone_schema.validate(abalone_raw, lazy=True)

    return abalone_validated


abalone = load_and_validate_abalone()

#Saving the validated dataset
abalone.to_csv("data/abalone_validated.csv", index=False)

# Peeking the dataframe
abalone


# Excluding column: Sex for cleaner display of graphs
new_column_names = ["Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

# Plot all variables against one another for EDA
chart = alt.Chart(abalone,width=150, height=100).mark_point().encode(
    alt.X(alt.repeat('row'), type='quantitative'),
    alt.Y(alt.repeat('column'), type='quantitative'),
    color=alt.Color("Sex:N", title="Abalone Sex")
).repeat(column = new_column_names, row = new_column_names
).properties(title = "Scatterplot matrix of abalone physical features and rings")

chart


# 1. One-hot encode the Sex categorical variable
# get_dummies() converts categories (M, F, I) -> columns Sex_F and Sex_M
# drop_first=True avoids creating redundant dummy columns

abalone_converted = pd.get_dummies(abalone, columns=["Sex"], drop_first=True)

# 2. Split predictors (X) and target variable (y)
# "Rings" is the target we want to predict (continuous -> regression)

X = abalone_converted.drop("Rings", axis=1)
y = abalone_converted["Rings"]
abalone_converted.head()

# 3. Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# validate training data for anomalous correlations between target/response variable 
# and features/explanatory variables, 
# as well as anomalous correlations between features/explanatory variables
# Do these on training data as part of EDA

# Combine X_train and y_train to form training dataset
abalone_train = X_train.copy()
abalone_train["Rings"] = y_train


# Setting up Dataset object for Deepcheck
abalone_train_ds = Dataset(
    abalone_train.drop(columns=["Sex_I", "Sex_M"]), 
    label="Rings",
    cat_features=[]
)

# 1. Feature–Label Correlation Check
#    Ensures no single feature is too predictive of the target.
#    PPS (predictive power score) must remain < 0.9.
check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=abalone_train_ds)

# 2. Feature–Feature Correlation Check
#    Ensures no pair of features has correlation above 0.99.
#    n_pairs = 0 means absolutely no correlated pairs allowed.
check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
    threshold = 0.99,
    n_pairs = 0
)
check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=abalone_train_ds)

# 3. Target Distribution Check (Label Drift)
#    Ensures the distribution of the target ("Rings") looks normal,
#    and that the training and test sets come from the same population.
#    This prevents unexpected label shifts.

# Build Deepchecks dataset for test data
abalone_test = X_test.copy()
abalone_test["Rings"] = y_test

abalone_test_ds = Dataset(
    abalone_test.drop(columns=["Sex_I", "Sex_M"]),
    label="Rings",
    cat_features=[]
)

# Run the target distribution drift check
check_label_drift = LabelDrift().add_condition_drift_score_less_than(0.2)
check_label_drift_result = check_label_drift.run(
    train_dataset=abalone_train_ds,
    test_dataset=abalone_test_ds
)


if not check_feat_lab_corr_result.passed_conditions():
    raise ValueError("Feature–Label correlation exceeds the acceptable threshold.")

if not check_feat_feat_corr_result.passed_conditions():
    raise ValueError("Feature–Feature correlation exceeds the acceptable threshold.")

if not check_label_drift_result.passed_conditions():
    raise ValueError("Target variable distribution drift detected ")



# 4. Standardize numeric features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# 5. Define the k-NN Regressor model
# weights="uniform" -> all neighbors contribute equally
# metric="minkowski", p=2 -> Euclidean distance
knn = KNeighborsRegressor(
    n_neighbors = 5,
    weights = "uniform",
    metric = "minkowski",
    p=2
)

# 6. Make predictions on both train and test sets
knn.fit(X_train_scaled, y_train)
y_train_pred = knn.predict(X_train_scaled)
y_test_pred  = knn.predict(X_test_scaled)

# 7. Evaluate performance using RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("=== KNN Regression performance (k=5) ===")
print(f"Train RMSE : {train_rmse:.4f}")
print(f"Test  RMSE : {test_rmse:.4f}")


#8. Plot Actual vs Predicted Rings for visual inspection
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_test_pred, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="black",
    label="Perfect Prediction Line"
)

plt.xlabel("Actual number of rings (True age proxy)")
plt.ylabel("Predicted number of rings")
plt.title("KNN model performance: actual vs predicted rings")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


