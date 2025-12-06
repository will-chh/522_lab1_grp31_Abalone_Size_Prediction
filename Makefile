.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: ## runs the targets: cl, env, build
	make cl
	make env
	make build

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \
		-p linux-aarch64

.PHONY: env
env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

.PHONY: run
run: ## alias for the up target
	make up

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

.PHONY: stop
stop: ## stop docker-compose services
	docker-compose stop

# docker multi architecture build rules -----
# image_name needs to be lowercase
DOCKER_USER=willchh
IMAGE_NAME=$(shell basename $(CURDIR) | tr '[:upper:]' '[:lower:]')

.PHONY: docker-build-push
docker-build-push: ## Build and push multi-arch image to Docker Hub (amd64 + arm64)
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag $(DOCKER_USER)/$(IMAGE_NAME):latest \
		--tag $(DOCKER_USER)/$(IMAGE_NAME):local-$(shell git rev-parse --short HEAD) \
		--push \
		.

.PHONY: docker-build-local
docker-build-local: ## Build single-arch image for local testing (current platform only)
	docker build \
		--tag $(DOCKER_USER)/$(IMAGE_NAME):local \
		.

# --------------------
# Analysis pipeline
# --------------------

# Step 1: data import
data/processed/cleaned_abalone.csv: utils/01_data_import.py
	python utils/01_data_import.py \
	  --input_path https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data \
	  --output_path data/processed/cleaned_abalone.csv

# Step 2: EDA
results/eda_scatter_matrix.png: data/processed/cleaned_abalone.csv utils/02_data_eda.py
	python utils/02_data_eda.py \
	  --input_path data/processed/cleaned_abalone.csv \
	  --output_path results/eda_scatter_matrix.png

# Step 3: train/test split
data/processed/train.csv data/processed/test.csv: data/processed/cleaned_abalone.csv utils/03_model_preprocess.py
	python utils/03_model_preprocess.py \
	  --input_path data/processed/cleaned_abalone.csv \
	  --train_output data/processed/train.csv \
	  --test_output data/processed/test.csv

# Step 4: model fitting
results/knn_model.pkl results/knn_scaler.pkl: data/processed/train.csv utils/04_model_fit.py
	python utils/04_model_fit.py \
	  --train_path data/processed/train.csv \
	  --model_output results/knn_model.pkl \
	  --scaler_output results/knn_scaler.pkl \
	  --n_neighbors 5

# Step 5: model evaluation (with plot)
results/knn_eval_plot.png: data/processed/train.csv data/processed/test.csv results/knn_model.pkl results/knn_scaler.pkl utils/05_model_eval.py
	python utils/05_model_eval.py \
	  --train_path data/processed/train.csv \
	  --test_path data/processed/test.csv \
	  --model_path results/knn_model.pkl \
	  --scaler_path results/knn_scaler.pkl \
	  --plot_output results/knn_eval_plot.png

# High-level pipeline target
.PHONY: analysis
analysis: results/knn_eval_plot.png ## Run full analysis pipeline (import → EDA → split → fit → eval)

# --------------------
# Step-by-step aliases
# --------------------

.PHONY: 01_data_import
01_data_import: data/processed/cleaned_abalone.csv ## Run only data import step

.PHONY: 02_data_eda
02_data_eda: results/eda_scatter_matrix.png ## Run only EDA step

.PHONY: 03_model_preprocess
03_model_preprocess: data/processed/train.csv data/processed/test.csv ## Run only model preprocess step

.PHONY: 04_model_fit
04_model_fit: results/knn_model.pkl results/knn_scaler.pkl ## Run only model fit step

.PHONY: 05_model_eval
05_model_eval: results/knn_eval_plot.png ## Run only model evaluation step
