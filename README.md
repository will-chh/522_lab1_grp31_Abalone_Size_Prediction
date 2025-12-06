# 522_lab1_grp31_Abalone_Size_Prediction
DSCI 522 Section 2 Group 31 Repository

# Abalone Abalone, Yummy Yummy in my Tummy! >< 

Welcome to our Abalone Size Prediction Project! 

## Description
This project aims to predict the age of an abalone from its physical features and sex. The model used in analysis is k-Nearest Neighbhours (k-NN) Regressor. The resulting model estimates the age of new abalone by identifying the k nearest abalones in the training set. 

Our final model results in a Test RMSE = 2.2884 in comparison to the Train RMSE = 1.8626 (with k = 5). 

The dataset in this project was obtained from the UCI Machine Learning Repository by Warwick Nash et al. and can be found at the link below.

https://archive.ics.uci.edu/dataset/1/abalone 

## Instructions
First the repository must be cloned locally using `git clone` <this repo> 

Next in your terminal navigate to the root directory of this project and run `jupyter lab` to open the project notebook.

All dependencies require can be found in the `environment.yml` file and must be installed before running the .ipynb notebook.

Install the project environment by running `conda env create --file environment.yml`

Lastly navigate to the Milestone1_Abalone_Age_Prediction.ipynb and run all cells. 

## Loading Docker Container: 
Commands Draft: 

#### Build the Docker image locally
docker build --no-cache -t willchh/522_grp31_abalone_age_prediction:latest .

#### Run the container interactively with a shell
docker run -it --rm \
 -p 8888:8888 \
 -v $(pwd):/workplace \
 -w /workplace \
 willchh/522_grp31_abalone_age_prediction:latest \
 bash

#### Run your analysis script manually
python script.py

## Running Scripts Separately
#### Step 1: Running only data_import
python utils/data_import.py \
  --input_path https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data \
  --output_path data/processed/cleaned_abalone.csv

#### Step 2: Running only data eda
python utils/data_eda.py \
  --input_path data/processed/cleaned_abalone.csv \
  --output_path results/eda_scatter_matrix.png

#### Step 3: Running model preprocess
python utils/model_preprocess.py \
  --input_path data/processed/cleaned_abalone.csv \
  --train_output data/processed/train.csv \
  --test_output data/processed/test.csv

#### Step 4: Running model eval
python utils/model_fit.py \
  --train_path data/processed/train.csv \
  --model_output results/knn_model.pkl \
  --scaler_output results/knn_scaler.pkl \
  --n_neighbors 5

#### Step 5: Model Evaluation Step with plotting Actual vs Predicted Values
python utils/model_eval.py \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --model_path results/knn_model.pkl \
  --scaler_path results/knn_scaler.pkl \
  --plot_output results/knn_eval_plot.png


#### Reminder for myself to put these in the Makefile:
import:
    python utils/data_import.py \
        --input_path https://archive.ics.uci.edu/... \
        --output_path data/processed/cleaned_abalone.csv

eda:
    python utils/data_eda.py \
        --input_path data/processed/cleaned_abalone.csv \
        --output_path results/eda_scatter_matrix.png

## Contributors:
- Yuting Ji

- Gurveer Madurai

- Seungmyun Park

- William Chong

## License 
MIT License
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license
