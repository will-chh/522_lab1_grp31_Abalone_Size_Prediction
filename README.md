# 522_lab1_grp31_Abalone_Size_Prediction
DSCI 522 Section 2 Group 31 Repository 

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

Lastly navigate to the Milestone1_XXX_Prediction.ipynb and run all cells. 

## Contributors:
- Yuting Ji

- Gurveer Madurai

- Seungmyun Park

- William Chong

## License 
MIT License