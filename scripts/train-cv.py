from sklearnex import patch_sklearn

patch_sklearn()

import os
import time
import json
import pprint
import logging
import argparse

from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from paus import AlgorithmEnum, set_seeds, load_dataset, create_model

GLOBAL_RANDOM_SEED = 42

# ----------------------------------------------------------------------------
#  CROSS-VALIDATION
# ----------------------------------------------------------------------------


class CliArguments(BaseModel):
    train_file: str
    params_file: str
    metrics_file: str

    cv: int
    algorithm: AlgorithmEnum


def main(args: CliArguments):
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    X, y = load_dataset(args.train_file)

    logger.info("Loading params...")
    with open(args.params_file, "r") as f:
        params = json.load(f)

    logger.info("Start cross-validation...")
    cv = StratifiedKFold(
        shuffle=True,
        n_splits=args.cv,
        random_state=GLOBAL_RANDOM_SEED,
    )

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info(f"Training fold {fold_i + 1}")

        # split data
        X_train, X_test = (
            X.iloc[train_idx],
            X.iloc[test_idx],
        )
        y_train, y_test = (
            y.iloc[train_idx],
            y.iloc[test_idx],
        )

        logger.info("Creating model...")
        model = create_model(args.algorithm, params)

        logger.info("Training...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_elapsed = time.time() - train_start

        logger.info("Evaluate on test data...")
        test_start = time.time()
        y_pred = model.predict(X_test)
        test_elapsed = time.time() - test_start

        metrics = {
            "algorithm": args.algorithm.value,
            "training_time": train_elapsed,
            "inference_time": test_elapsed,
            "mcc": matthews_corrcoef(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

        logger.info("Metrics:")
        pprint.pprint(metrics)
        print(classification_report(y_test, y_pred))

        logger.info("Saving metrics...")
        with open(args.metrics_file, "a+") as f:
            json.dump(metrics, f)
            f.write("\n")


# ----------------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    # initialize
    set_seeds()

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, choices=[x.value for x in AlgorithmEnum])

    # -- inputs
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--params-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)

    parser.add_argument("--cv", type=int, default=10)

    # start app
    opts = CliArguments(**vars(parser.parse_args()))
    print(opts)

    try:
        main(opts)
    except KeyboardInterrupt:
        print("Aborted")
