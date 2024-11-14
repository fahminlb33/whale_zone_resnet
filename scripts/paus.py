import random
from enum import Enum

import numpy as np
import pandas as pd
import tensorflow as tf

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

DROP_COLS = ["zone_id", "ts", "target", "country", "continent"]


class AlgorithmEnum(Enum):
    # scikit-learn
    KNN = "knn"
    LOGISTIC_REGRESSION = "logistic-regression"
    DECISION_TREE = "decision-tree"
    RANDOM_FOREST = "random-forest"
    GRADIENT_BOOSTING = "gradient-boosting"

    # xgboost
    XGBOOST = "xgboost"

    # tensorflow
    NN_VANILLA = "ann"
    NN_RESNET = "resnet"


class NeuralNetClassifier:
    def __init__(
        self,
        mode: str,
        epochs=100,
        batch_size=512,
        depth=1,
        hidden_units=128,
        loss="bce",
        learning_rate=1e-3,
        verbose=2,
    ):
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.depth = depth
        self.hidden_units = hidden_units
        self.loss = loss
        self.verbose = verbose
        self.learning_rate = learning_rate

    @staticmethod
    def resblock(x, size):
        fx = tf.keras.layers.Dense(size, activation="relu")(x)
        fx = tf.keras.layers.BatchNormalization()(fx)
        fx = tf.keras.layers.Dense(size, activation="relu")(fx)
        out = tf.keras.layers.Add()([x, fx])
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.BatchNormalization()(out)

        return out

    def create_model(self, train_data):
        # create input layer
        inputs = tf.keras.Input(shape=(train_data.shape[1],))

        # create preprocessing layer
        norm_layer = tf.keras.layers.Normalization()
        norm_layer.adapt(train_data)

        x = norm_layer(inputs)

        # create hidden layers
        if self.mode == "ann":
            for _ in range(self.depth):
                x = tf.keras.layers.Dense(self.hidden_units, activation="relu")(x)
        elif self.mode == "resnet":
            x = tf.keras.layers.Dense(self.hidden_units, activation="relu")(x)
            for _ in range(self.depth):
                x = NeuralNetClassifier.resblock(x, self.hidden_units)
        else:
            raise ValueError(f"Invalid kind: {self.mode}")

        # create output layer
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # create model
        self.norm_layer = norm_layer
        self.model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name=f"model_{self.mode}"
        )

        # create loss
        loss_func = (
            tf.keras.losses.BinaryCrossentropy(from_logits=False)
            if self.loss == "bce"
            else tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False)
        )

        # compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_func,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="prc", curve="PR"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
            ],
        )

        print(self.model.summary())

    def fit(self, X, y, validation_data=None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            y = np.asarray(y)

        if validation_data is not None and not isinstance(validation_data[0], np.ndarray):
            validation_data = (np.asarray(validation_data[0]), np.asarray(validation_data[1]))

        # input_shape = (ds_train.element_spec[0].shape[1],)
        self.create_model(X)

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            validation_data=validation_data,
            verbose=self.verbose,
        )

    def predict(self, X):
        y_pred = self.model.predict(X, verbose=2)
        return np.where(y_pred > 0.5, 1, 0).ravel()
    
    def predict_proba(self, X):
        y_pred = self.model.predict(X, verbose=2)
        return np.array((1.0 - y_pred, y_pred))
    
    def get_total_params(self) -> tuple[int, int]:        
        return (
            int(sum(np.prod(p) for p in [v.shape for v in self.model.trainable_weights])),
            int(sum(np.prod(p) for p in [v.shape for v in self.model.non_trainable_weights])),
        )


def set_seeds(RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def load_dataset(path: str):
    # load dataset
    df = pd.read_parquet(path)

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df["target"]

    return X, y


def create_model(algorithm: AlgorithmEnum, params: dict):
    # scikit-learn
    if algorithm == AlgorithmEnum.KNN:
        return KNeighborsClassifier(**params)
    elif algorithm == AlgorithmEnum.LOGISTIC_REGRESSION:
        return LogisticRegression(**params)
    elif algorithm == AlgorithmEnum.RANDOM_FOREST:
        return RandomForestClassifier(**params)
    elif algorithm == AlgorithmEnum.DECISION_TREE:
        return DecisionTreeClassifier(**params)
    elif algorithm == AlgorithmEnum.GRADIENT_BOOSTING:
        return HistGradientBoostingClassifier(**params)

    # xgboost
    elif algorithm == AlgorithmEnum.XGBOOST:
        return XGBClassifier(**params)

    # tensorflow
    elif algorithm == AlgorithmEnum.NN_VANILLA:
        return NeuralNetClassifier(mode="ann", **params)
    elif algorithm == AlgorithmEnum.NN_RESNET:
        return NeuralNetClassifier(mode="resnet", **params)
    
    raise ValueError("Unknown algorithm!")
