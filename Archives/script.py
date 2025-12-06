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

from config import DATA_URL, VALIDATED_PATH
from utils.data_import import load_and_validate_abalone
from utils.data_eda import scatter_matrix
from utils.model_preprocess import preprocess_and_split
from utils.model_fit import fit_knn_regressor
from utils.model_eval import evaluate_knn

#This ensures our data is freshly loaded everytime :) 
#Creating a function here for the loading process: 
abalone = load_and_validate_abalone(DATA_URL)

#Saving the validated dataset
abalone.to_csv(VALIDATED_PATH, index=False)

# Peeking the dataframe
abalone

# Plot all variables against one another for EDA
chart = scatter_matrix(abalone)

chart

# preprocess and split the data
X_train, X_test, y_train, y_test = preprocess_and_split(abalone)

# get knn model and scaler
knn, scaler = fit_knn_regressor(X_train, y_train, n_neighbors=5)

# evaluate the model
evaluate_knn(knn, scaler, X_train, y_train, X_test, y_test, plot=True)