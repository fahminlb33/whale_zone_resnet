#!/usr/bin/env bash

export PYTHONUNBUFFERED="1"

# python scripts/importance.py xgboost \
#     --train-file ./dataset/paper2-train.parquet \
#     --test-file ./dataset/paper2-test.parquet \
#     --params-file ./params/best_params_xgboost.json \
#     --output-path ./results/feature-importance

# python scripts/importance.py gradient-boosting \
#     --train-file ./dataset/paper2-train.parquet \
#     --test-file ./dataset/paper2-test.parquet \
#     --params-file ./params/best_params_gradient_boosting.json \
#     --output-path ./results/feature-importance

# python scripts/importance.py random-forest \
#     --train-file ./dataset/paper2-train.parquet \
#     --test-file ./dataset/paper2-test.parquet \
#     --params-file ./params/best_params_random_forest.json \
#     --output-path ./results/feature-importance

# python scripts/importance.py decision-tree \
#     --train-file ./dataset/paper2-train.parquet \
#     --test-file ./dataset/paper2-test.parquet \
#     --params-file ./params/best_params_decision_tree.json \
#     --output-path ./results/feature-importance

# python scripts/importance.py logistic-regression \
#     --train-file ./dataset/paper2-train.parquet \
#     --test-file ./dataset/paper2-test.parquet \
#     --params-file ./params/best_params_logistic_regression.json \
#     --output-path ./results/feature-importance

python scripts/importance.py knn \
    --train-file ./dataset/paper2-train.parquet \
    --test-file ./dataset/paper2-test.parquet \
    --params-file ./params/best_params_knn.json \
    --output-path ./results/feature-importance

