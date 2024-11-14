from sklearnex import patch_sklearn

patch_sklearn()

import os
import json
import logging
import argparse
import multiprocessing

import tqdm
import shap
import matplotlib
import numpy as np
import pandas as pd

from pydantic import BaseModel
from sklearn.metrics import    matthews_corrcoef

from paus import AlgorithmEnum, set_seeds, load_dataset, create_model

GLOBAL_RANDOM_SEED = 42

# ----------------------------------------------------------------------------
#  FEATURE IMPORTANCE
# ----------------------------------------------------------------------------

# https://stackoverflow.com/questions/77258642/feature-importance-keras-regressionmodel
def permutation_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, algorithm: str, output_path: str, n_permutations=5):
    # get baseline performance
    y_pred = model.predict(X_test)
    baseline_score = matthews_corrcoef(y_test, y_pred)

    # permute each features
    for feature_name in tqdm.tqdm(X_test.columns):
        # permute n times
        for shuffle_i in range(n_permutations):
            # clone dataset
            X_copy = X_test.copy()

            # permute feature (i.e. shuffle)
            X_copy[feature_name] = X_copy[feature_name].sample(frac=1).values

            # eval new performance
            y_pred = model.predict(X_copy)
            permute_score = matthews_corrcoef(y_test, y_pred)

            # save metric
            with open(os.path.join(output_path, "metrics-permute.jsonl"), "a+") as f:
                json.dump({
                    "algorithm": algorithm,
                    "permute_index": shuffle_i,
                    "permuted_feature": feature_name,
                    "baseline_score": baseline_score,
                    "permute_score": permute_score,
                    "total_loss": baseline_score - permute_score,
                }, f)
                f.write("\n")

# https://github.com/shap/shap/issues/632#issuecomment-569286406
def shap_importance(model, X_train: pd.DataFrame, X_test: pd.DataFrame, algorithm: str, output_path: str):
    # create explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(n=100, random_state=GLOBAL_RANDOM_SEED))

    # get shap values
    shap_values = explainer.shap_values(X_test.sample(n=100, random_state=GLOBAL_RANDOM_SEED))

    # get mean average feature importance
    mean_importance = np.abs(shap_values).mean(axis=0).sum(axis=1)
    
    with open(os.path.join(output_path, "metrics-shap.jsonl"), "a+") as f:
        json.dump({
            "algorithm": algorithm,
            "shap_values": {k:v for k, v in zip(X_train.columns, mean_importance)}
        }, f)
        f.write("\n")

class CliArguments(BaseModel):
    train_file: str
    test_file: str
    params_file: str
    output_path: str

    algorithm: AlgorithmEnum


def main(args: CliArguments):
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    X_train, y_train = load_dataset(args.train_file)
    X_test, y_test = load_dataset(args.test_file)

    logger.info("Loading params...")
    with open(args.params_file, "r") as f:
        params = json.load(f)

    logger.info("Creating model...")
    model = create_model(args.algorithm, params)

    logger.info("Training...")
    model.fit(X_train, y_train)

    logger.info("Pemutation importance analysis...")
    permutation_feature_importance(model, X_test, y_test, args.algorithm.value, args.output_path)

    logger.info("SHAP analysis...")
    shap_importance(model, X_train, X_test, args.algorithm.value, args.output_path)



if __name__ == "__main__":
    # initialize
    logging.basicConfig(level=logging.INFO)
    set_seeds()
    matplotlib.use("Agg")

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, choices=[x.value for x in AlgorithmEnum])

    # -- inputs
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--params-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    # start app
    opts = CliArguments(**vars(parser.parse_args()))
    print(opts)

    try:
        main(opts)
    except KeyboardInterrupt:
        print("Aborted")
