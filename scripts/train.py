import time
import json
import pprint
import random
import argparse

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

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

EPOCHS = 200
BATCH_SIZE = 256


def resblock(x, size):
	fx = tf.keras.layers.Dense(size, activation='relu')(x)
	fx = tf.keras.layers.BatchNormalization()(fx)
	fx = tf.keras.layers.Dense(size, activation='relu')(fx)
	out = tf.keras.layers.Add()([x, fx])
	out = tf.keras.layers.ReLU()(out)
	out = tf.keras.layers.BatchNormalization()(out)
	
	return out

def create_model(kind, depth, ds_train) -> tf.keras.Model:
	# create input layer
	inputs = tf.keras.Input(shape=(ds_train.element_spec[0].shape[1],))

	# create preprocessing layer
	norm_layer = tf.keras.layers.Normalization()
	norm_layer.adapt(ds_train.map(lambda x, _: x))

	x = norm_layer(inputs)

	# create hidden layers
	if kind == "fcn":
		for _ in range(depth):
			x = tf.keras.layers.Dense(128, activation='relu')(x)
	elif kind == "resnet":
		x = tf.keras.layers.Dense(128, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.Ones())(x)
		for _ in range(depth):
			x = resblock(x, 128)
	elif kind == "resnet_bias":
		x = tf.keras.layers.Dense(128, activation="relu")(x)
		for _ in range(depth):
			x = resblock(x, 128)
	else:
		raise ValueError(f"Invalid kind: {kind}")

	# create output layer
	outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

	# create model
	return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"model_{kind}")

def create_loss(lf):
	if lf == "bce":
		return tf.keras.losses.BinaryCrossentropy(from_logits=False)
	
	return tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False)

def main(args):
	model_file_name = f"{args.dataset_name}_{args.model}_{args.loss}_{args.depth}"

	print("Loading data...")
	arr_train = pd.read_parquet(f"../dataset/train-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values
	arr_test = pd.read_parquet(f"../dataset/test-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values
	arr_val = pd.read_parquet(f"../dataset/validation-{args.dataset_name}.parquet").drop(columns=["country", "zone_id", "ts"], errors="ignore").values

	print("Creating tf.data...")
	ds_train = tf.data.Dataset.from_tensor_slices((arr_train[:, :-1], arr_train[:, -1])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	ds_val = tf.data.Dataset.from_tensor_slices((arr_val[:, :-1], arr_val[:, -1])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	
	print("Creating model...")
	model = create_model(args.model, args.depth, ds_train)

	print("Compiling model...")
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss=create_loss(args.loss),
		metrics=[
			tf.keras.metrics.TruePositives(name='tp'),
			tf.keras.metrics.FalsePositives(name='fp'),
			tf.keras.metrics.TrueNegatives(name='tn'),
			tf.keras.metrics.FalseNegatives(name='fn'),
			tf.keras.metrics.BinaryAccuracy(name='accuracy'),
			tf.keras.metrics.Precision(name='precision'),
			tf.keras.metrics.Recall(name='recall'),
			tf.keras.metrics.AUC(name='auc'),
			tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
	])

	print(model.summary())
	
	print("Creating callbacks...")
	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor='val_prc',
		verbose=1,
		patience=10,
		mode='max',
		restore_best_weights=True)

	log_dir = f"logs/{model_file_name}"
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	print("Training...")
	train_start = time.time()
	model.fit(
		ds_train,
		epochs=EPOCHS,
		validation_data=ds_val,
		verbose=2,
		callbacks=[tensorboard_callback, early_stopping])
	train_elapsed = time.time() - train_start
	
	print("Evaluate on test data")
	X_test = arr_test[:, :-1]
	y_test = arr_test[:, -1].astype(int)

	test_start = time.time()
	y_pred = model.predict(X_test, verbose=2)
	y_pred = np.where(y_pred > 0.5, 1, 0).ravel()
	test_elapsed = time.time() - test_start

	metrics = {
		"params_model": args.model, 
		"params_loss": args.loss, 
		"params_depth": args.depth, 
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
	model.save(f"models/{model_file_name}.keras")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="autogbifml")
	parser.add_argument("-m", "--model", choices=["fcn", "resnet", "resnet_bias"])
	parser.add_argument("-l", "--loss", choices=["bce", "fl"])
	parser.add_argument("-d", "--depth", type=int)
	parser.add_argument("-n", "--dataset-name")

	args = parser.parse_args()
	print(repr(args))
	
	main(args)
