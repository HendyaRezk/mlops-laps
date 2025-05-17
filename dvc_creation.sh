#!/bin/bash

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    git commit -m "Initialize DVC"
fi

# Move config to proper location (fixes Hydra path issues)
echo "Setting up config files..."
mkdir -p src/conf
cp conf/config.yaml src/conf/
git add src/conf/config.yaml
git commit -m "Move config to src/conf"

# Add data files to DVC tracking
echo "Tracking data files with DVC..."
dvc add data/raw/train.csv
dvc add data/raw/X_test.csv
dvc add data/raw/y_test.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track raw data files"

# Create DVC pipeline
echo "Creating DVC pipeline..."
cat > dvc.yaml << 'EOL'
stages:
  preprocess:
    cmd: python src/data/preprocess_data.py
    wdir: .
    deps:
      - src/data/preprocess_data.py
      - data/raw/train.csv
    outs:
      - data/interim/feature_Eng_data.csv
      - data/processed/X_train.csv
      - data/processed/y_train.csv
      - data/processed/preprocessor.pkl
    params:
      - config.data

  train:
    cmd: python src/model/train_model.py
    wdir: .
    deps:
      - src/model/train_model.py
      - data/processed/X_train.csv
      - data/processed/y_train.csv
      - data/processed/preprocessor.pkl
    outs:
      - src/model/model_only.pkl
      - src/model/full_pipeline_model.pkl
    params:
      - config.model

  evaluate:
    cmd: python src/evaluate.py
    wdir: .
    deps:
      - src/evaluate.py
      - src/model/full_pipeline_model.pkl
      - data/raw/X_test.csv
      - data/raw/y_test.csv
    metrics:
      - reports/metrics.json:
          cache: false
    params:
      - config.evaluation
EOL

# Create params file
echo "Creating params.yaml..."
cat > params.yaml << 'EOL'
config:
  data:
    raw_path: "data/raw/train.csv"
    interim_path: "data/interim/feature_Eng_data.csv"
    processed_path: "data/processed"
    x_test_path: "data/raw/X_test.csv"
    y_test_path: "data/raw/y_test.csv"
    preprocessor_pkl: "data/processed/preprocessor.pkl"
  
  model:
    random_forest:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
    model_only_path: "src/model/model_only.pkl"
    combined_model_path: "src/model/full_pipeline_model.pkl"
    
  evaluation:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
EOL

# Set up remote storage (example: local)
echo "Configuring remote storage..."
mkdir -p /tmp/dvc-storage
dvc remote add -d localremote /tmp/dvc-storage

# Commit pipeline
echo "Committing pipeline..."
git add dvc.yaml params.yaml
git commit -m "Initialize DVC pipeline"

echo "DVC setup complete!"
echo "Run the pipeline with: dvc repro"