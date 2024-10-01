import os
import glob
import time
import random
import pprint
import itertools
import subprocess

import numpy as np
import pandas as pd

import tensorflow as tf
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
tf.random.set_seed(RANDOM_SEED)

BATCH_SIZE = 256

features = [
    "pbo_sum",
    "pbo_mean",
    "tob_mean",
    "sob_sum",
    "tob_sum",
    "sob_mean",
    "fe_mean",
    "fe_sum",
    "so_sum",
    "po4_mean",
    # "target",
]

# load dataset
df_test = pd.read_parquet(f"../dataset/test-all.parquet")

y_test = df_test["target"].values
X_test_sel = df_test[features].values
X_test_all = df_test.drop(columns=["country", "zone_id", "ts", "target"], errors="ignore").values

# eval on test data
eval_results = []
for model_path in glob.glob("./models/*.keras"):
    filename = os.path.basename(model_path)
    print("Eval:", filename)

    try:
        # load model
        model = tf.keras.models.load_model(model_path)

        # perform prediction
        start_test_time = time.time()
        if "sel-10" in model_path:
            y_pred = (model.predict(X_test_sel, batch_size=BATCH_SIZE) > 0.5).ravel()
        else:
            y_pred = (model.predict(X_test_all, batch_size=BATCH_SIZE) > 0.5).ravel()
        test_elapsed = time.time() - start_test_time

        # eval
        eval_results.append({
            "filename": filename,
            "inference_time": test_elapsed,
            "mcc": matthews_corrcoef(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
            "ap": average_precision_score(y_test, y_pred),
        })

        print(eval_results[-1])
    except Exception as e:
        print("ERROR!", model_path, e)
    
# create dataframe
df_result = pd.DataFrame(eval_results)
print(df_result.head())

df_result.to_csv("results.csv", index=False)
