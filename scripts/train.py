import argparse
import datetime
import itertools

import numpy as np
import pandas as pd

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

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

def create_model(kind, ds_train) -> tf.keras.Model:
	# create input layer
	inputs = tf.keras.Input(shape=(ds_train.element_spec[0].shape[1],))

	# create preprocessing layer
	norm_layer = tf.keras.layers.Normalization()
	norm_layer.adapt(ds_train.map(lambda x, _: x))

	x = norm_layer(inputs)
	# x = inputs

	# create hidden layers
	if kind == "fcn":
		x = tf.keras.layers.Dense(128, activation='relu')(x)
		x = tf.keras.layers.Dense(128, activation='relu')(x)
		x = tf.keras.layers.Dense(128, activation='relu')(x)

		x = tf.keras.layers.Dense(128, activation='relu')(x)
		x = tf.keras.layers.Dense(128, activation='relu')(x)
		x = tf.keras.layers.Dense(128, activation='relu')(x)
	elif kind == "resnet":
		x = tf.keras.layers.Dense(128, activation='relu')(x)
		x = resblock(x, 128)
		x = resblock(x, 128)
		x = resblock(x, 128)

		x = resblock(x, 128)
		x = resblock(x, 128)
		x = resblock(x, 128)
	else:
		raise ValueError(f"Invalid kind: {kind}")

	# create output layer
	outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

	# create model
	return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"model_{kind}")

def create_loss(lf):
	if lf == "bc":
		return tf.keras.losses.BinaryCrossentropy(from_logits=False)
	
	return tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False)

def main(args):
	loss_name = args.loss
	model_name = args.model
	dataset_suffix = args.dataset

	print("Loading data...")
	arr_train = pd.read_parquet(f"../dataset/train-{dataset_suffix}.parquet").values
	arr_test = pd.read_parquet(f"../dataset/test-{dataset_suffix}.parquet").values
	arr_val = pd.read_parquet(f"../dataset/validation-{dataset_suffix}.parquet").values

	print("Creating tf.data...")
	ds_train = tf.data.Dataset.from_tensor_slices((arr_train[:, :-1], arr_train[:, -1])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	ds_val = tf.data.Dataset.from_tensor_slices((arr_val[:, :-1], arr_val[:, -1])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	ds_test = tf.data.Dataset.from_tensor_slices((arr_test[:, :-1], arr_test[:, -1])).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	
	print("Creating model...")
	model = create_model(model_name, ds_train)

	print("Compiling model...")
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss=create_loss(loss_name),
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

	log_dir = f"logs/fit/model-{dataset_suffix}_{model_name}_{loss_name}"
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	print("Training...")
	model.fit(
		ds_train,
		epochs=EPOCHS,
		validation_data=ds_val,
		callbacks=[early_stopping, tensorboard_callback])
	
	print("Evaluate on test data")
	results = model.evaluate(ds_test)
	print("test loss, test acc:", results)
	
	print("Saving model...")
	model.save(f"models/model-{dataset_suffix}_{model_name}_{loss_name}.keras")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="autogbifml")
	parser.add_argument("-m", "--model")
	parser.add_argument("-l", "--loss")
	parser.add_argument("-d", "--dataset")

	args = parser.parse_args()
	
	main(args)
