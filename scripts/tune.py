from sklearnex import patch_sklearn

patch_sklearn()

import os
import json
import argparse
import multiprocessing

import mlflow
import optuna
import numpy as np

from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
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
#  OPTUNA HYPERPARAMETER OPTIMIZATION
# ----------------------------------------------------------------------------


class OptunaObjectiveOptions(BaseModel):
    dataset_file: str

    cv: int
    jobs: int
    algorithm: AlgorithmEnum


class OptunaObjective:
    def __init__(self, config: OptunaObjectiveOptions) -> None:
        self.args = config
        self.load_data()

    def load_data(self):
        self.X, self.y = load_dataset(self.args.dataset_file)

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create parameters
            trial_params = self.get_trial_params(trial)

            # log params to mlflow
            mlflow.log_params(trial_params)

            # create scores
            scores = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "mcc": [],
                "roc_auc": [],
            }

            # cross-validation settings
            cv = StratifiedKFold(
                shuffle=True,
                n_splits=self.args.cv,
                random_state=GLOBAL_RANDOM_SEED,
            )

            # perform cross-validation
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
                print(f">>> Training fold {fold_i + 1}")

                # split data
                X_train, X_test = (
                    self.X.iloc[train_idx],
                    self.X.iloc[test_idx],
                )
                y_train, y_test = (
                    self.y.iloc[train_idx],
                    self.y.iloc[test_idx],
                )

                # fit model
                clf = create_model(self.args.algorithm, trial_params)
                clf.fit(X_train, y_train)

                # run prediction
                y_pred = clf.predict(X_test)

                # log metrics
                scores["accuracy"].append(accuracy_score(y_test, y_pred))
                scores["precision"].append(precision_score(y_test, y_pred))
                scores["recall"].append(recall_score(y_test, y_pred))
                scores["f1"].append(f1_score(y_test, y_pred))
                scores["mcc"].append(matthews_corrcoef(y_test, y_pred))
                scores["roc_auc"].append(roc_auc_score(y_test, y_pred))

                if self.args.algorithm == AlgorithmEnum.NN_RESNET or self.args.algorithm == AlgorithmEnum.NN_VANILLA:
                    total_params = clf.get_total_params()
                    scores["trainable_params"] = total_params[0]
                    scores["non_trainable_params"] = total_params[1]

            # log metrics to mlflow
            for metric_name, metric_values in scores.items():
                mlflow.log_metric(metric_name, np.mean(metric_values))

            # return MCC score from CV to maximize
            return np.mean(scores["mcc"])

    def get_trial_params(self, trial: optuna.Trial) -> dict:
        if self.args.algorithm == AlgorithmEnum.KNN:
            return {
                # tunable params
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "metric": trial.suggest_categorical(
                    "metric", ["euclidean", "minkowski"]
                ),
                # fixed params
                "n_jobs": self.args.jobs,
            }

        elif self.args.algorithm == AlgorithmEnum.LOGISTIC_REGRESSION:
            return {
                # tunable params
                "C": trial.suggest_float("C", 1, 100),
                "solver": trial.suggest_categorical(
                    "solver", ["newton-cg", "lbfgs", "liblinear"]
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                # fixed params
                "n_jobs": self.args.jobs,
                "random_state": GLOBAL_RANDOM_SEED,
            }

        elif self.args.algorithm == AlgorithmEnum.RANDOM_FOREST:
            return {
                # tunable params
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
                "max_depth": trial.suggest_int("max_depth", 3, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                # fixed parameters
                "n_jobs": self.args.jobs,
                "random_state": GLOBAL_RANDOM_SEED,
            }

        elif self.args.algorithm == AlgorithmEnum.DECISION_TREE:
            return {
                # tunable params
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "max_features": trial.suggest_categorical(
                    "max_features", [None, "sqrt"]
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 26),
                # fixed parameters
                "random_state": GLOBAL_RANDOM_SEED,
            }

        elif self.args.algorithm == AlgorithmEnum.GRADIENT_BOOSTING:
            return {
                # tunable params
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 100),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                # fixed parameters
                "random_state": GLOBAL_RANDOM_SEED,
            }

        elif self.args.algorithm == AlgorithmEnum.XGBOOST:
            return {
                # tunable params
                "max_depth": trial.suggest_int("max_depth", 3, 110),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 1, log=True),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.01, 1.0, log=True
                ),
                # fixed parameters
                "device": "gpu",
                "tree_method": "hist",
                "random_state": GLOBAL_RANDOM_SEED,
            }

        elif (
            self.args.algorithm == AlgorithmEnum.NN_VANILLA
            or self.args.algorithm == AlgorithmEnum.NN_RESNET
        ):
            return {
                # tunable params
                "epochs": trial.suggest_int("epochs", 50, 200),
                "loss": trial.suggest_categorical("loss", ["bce", "fl"]),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "depth": trial.suggest_int("depth", 1, 6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.01, log=True),
                "hidden_units": trial.suggest_categorical(
                    "hidden_units", [8, 16, 32, 64, 128, 256, 512]
                ),
            }


# ----------------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------------


class CliArguments(BaseModel):
    dataset_file: str

    algorithm: AlgorithmEnum
    name: str
    cv: int
    trials: int
    storage: str
    tracking_url: str

    n_jobs: int


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        print(f"Found experiment: {experiment.experiment_id}")
        return experiment.experiment_id
    else:
        print(f"Creating experiment: {experiment_name}")
        return mlflow.create_experiment(experiment_name)


def main(args: CliArguments):
    # set mlflow tracking
    if args.tracking_url:
        print(f"Setting mlflow tracking url: {args.tracking_url}")
        mlflow.set_tracking_uri(args.tracking_url)

    # create objective
    objective = OptunaObjective(
        OptunaObjectiveOptions(
            dataset_file=args.dataset_file,
            algorithm=args.algorithm,
            cv=args.cv,
            jobs=args.n_jobs,
        )
    )

    # load dataset
    print("Loading dataset...")
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment(args.name)
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.name,
        storage=args.storage,
        load_if_exists=True,
    )

    # start optimization
    print(f"Starting optimization with {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials)

    # get best parameters
    print(json.dumps(study.best_params))
        


if __name__ == "__main__":
    # initialize
    set_seeds()

    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, choices=[x.value for x in AlgorithmEnum])

    # -- inputs
    parser.add_argument("--dataset-file", type=str, required=True)

    # -- tuning
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///tune.db",
    )
    parser.add_argument(
        "--tracking-url",
        type=str,
    )

    # -- misc
    parser.add_argument("--n-jobs", type=int, default=multiprocessing.cpu_count() - 2)

    # start app
    opts = CliArguments(**vars(parser.parse_args()))
    print(opts)

    try:
        main(opts)
    except KeyboardInterrupt:
        print("Aborted")
