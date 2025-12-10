import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = Path("data")
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"


def build_lipnet_style_model(input_shape, num_classes: int) -> tf.keras.Model:
    """
    LipNet-style model:
    - TimeDistributed CNN on each frame
    - Temporal modeling with Bidirectional LSTM
    - Dense + Softmax for final classification
    """
    inputs = layers.Input(shape=input_shape)  # (T, H, W, C)

    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation="relu", padding="same")
    )(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation="relu", padding="same")
    )(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False)
    )(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(
            f"Expected X.npy and y.npy in {DATA_DIR}. "
            "X shape: (N, T, H, W, C), y shape: (N,)."
        )

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    time_steps, height, width, channels = X_train.shape[1:]
    num_classes = len(np.unique(y))

    model = build_lipnet_style_model(
        input_shape=(time_steps, height, width, channels),
        num_classes=num_classes,
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=8,
    )

    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    model.save("lipnet_style_model.h5")
    print("Model saved to lipnet_style_model.h5")


if __name__ == "__main__":
    main()
