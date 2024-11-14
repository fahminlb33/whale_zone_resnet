from sklearnex import patch_sklearn

patch_sklearn()

import os
import time
import json
import pprint
import logging
import argparse
import multiprocessing

import matplotlib
from matplotlib.figure import Figure
from pydantic import BaseModel
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from paus import AlgorithmEnum, set_seeds, load_dataset, create_model

GLOBAL_RANDOM_SEED = 42

# ----------------------------------------------------------------------------
#  TRAIN AND EVALUATE
# ----------------------------------------------------------------------------


class CliArguments(BaseModel):
    train_file: str
    test_file: str
    params_file: str
    output_path: str

    algorithm: AlgorithmEnum

    n_jobs: int = multiprocessing.cpu_count() - 2


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
    train_start = time.time()
    model.fit(X_train, y_train)
    train_elapsed = time.time() - train_start

    logger.info("Evaluate on test data...")
    test_start = time.time()
    y_pred = model.predict(X_test)
    test_elapsed = time.time() - test_start

    metrics = {
        "algorithm": args.algorithm,
        "training_time": train_elapsed,
        "inference_time": test_elapsed,
        "mcc": matthews_corrcoef(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
    }

    pprint.pprint(metrics)
    print(classification_report(y_test, y_pred))

    logger.info("Metrics:")
    pprint.pprint(metrics)
    print(classification_report(y_test, y_pred))

    logger.info("Saving metrics...")
    with open(os.path.join(args.output_path, "metrics.jsonl"), "a+") as f:
        json.dump(metrics, f)
        f.write("\n")

    # plot confusion matrix
    fig = Figure()
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(fig, os.path.join(args.output_path, "confusion_matrix.png"))

    # plot ROC
    fig = Figure()
    ax = fig.subplots()
    RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(fig, os.path.join(args.output_path, "roc_curve.png"))

    # plot precision-recall
    fig = Figure()
    ax = fig.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(fig, os.path.join(args.output_path, "precision_recall.png"))


if __name__ == "__main__":
    # initialize
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
