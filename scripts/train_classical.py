import time
import json
import pprint
import random
import argparse

# from sklearnex import patch_sklearn
# patch_sklearn()

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    matthews_corrcoef, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score,
)

# repro
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_model(args):
	if args.model == "svm":
		return SVC(random_state=42, verbose=True)
	elif args.model == "dt":
		return DecisionTreeClassifier(random_state=42,)
	elif args.model == "rf":
		return RandomForestClassifier(random_state=42, max_depth=1000, verbose=True, n_jobs=4)
	elif args.model == "gb":
		return GradientBoostingClassifier(random_state=42, verbose=True)
	else:
		raise ValueError("Unknown model!")


def main(args):
	model_file_name = f"{args.dataset_name}_{args.model}"

	print("Loading data...")
	arr_train = pd.read_parquet(f"../dataset/zonal/train-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values
	arr_test = pd.read_parquet(f"../dataset/zonal/test-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values
	# arr_val = pd.read_parquet(f"../dataset/zonal/validation-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values
	
	print("Creating model...")
	model = create_model(args)

	X_train = arr_train[:, :-1]
	y_train = arr_train[:, -1].astype(int)

	print("Training...")
	train_start = time.time()
	model.fit(X_train, y_train)
	train_elapsed = time.time() - train_start
	
	print("Evaluate on test data")
	X_test = arr_test[:, :-1]
	y_test = arr_test[:, -1].astype(int)

	test_start = time.time()
	y_pred = model.predict(X_test)
	test_elapsed = time.time() - test_start

	metrics = {
		"params_model": args.model, 
		"params_dataset": args.dataset_name, 

		"training_time": train_elapsed,
		"inference_time": test_elapsed,
		"mcc": matthews_corrcoef(y_test, y_pred),
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred),
		"recall": recall_score(y_test, y_pred),
		"f1": f1_score(y_test, y_pred),
		"roc_auc": roc_auc_score(y_test, y_pred),
		"ap": average_precision_score(y_test, y_pred),
	}

	pprint.pprint(metrics)
	print(classification_report(y_test, y_pred))

	with open("metrics.jsonl", "a+") as f:
		json.dump(metrics, f)
		f.write("\n")
	
	print("Saving model...")
	# joblib.dump(model, f"models/{model_file_name}.joblib")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="autogbifml")
	parser.add_argument("-m", "--model", choices=["svm", "dt", "rf", "gb"])
	parser.add_argument("-n", "--dataset-name")

	args = parser.parse_args()
	print(repr(args))
	
	main(args)
