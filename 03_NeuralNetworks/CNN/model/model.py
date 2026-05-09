import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape input to 28x28x1 for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

# ---- OPTIMIZATION: slightly deeper conv blocks + batch normalization ----
model = Sequential(
    [
        Conv2D(32, 3, activation="relu", padding="same", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, 3, activation="relu", padding="same"),
        MaxPooling2D(2),
        Dropout(0.25),
        Conv2D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(2),
        Dropout(0.25),
        Flatten(),  # 2D to 1D
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
# ---- OPTIMIZATION: reduce learning rate when validation loss stalls ----
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5)

# ---- OPTIMIZATION: a few more epochs with early stopping ----
model.fit(
    x_train_cnn,
    y_train_cat,
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
)

model_path = os.path.join(os.path.dirname(__file__), "num_cnn_model.h5")
model.save(model_path)

test_loss, test_acc = model.evaluate(x_test_cnn, y_test_cat)
print("Test accuracy:", test_acc)

import numpy as np

# Predict the first 5 test samples
predictions = model.predict(x_test_cnn[:5])
for i in range(5):
    plt.imshow(x_test_cnn[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])} - True: {y_test[i]}")
    plt.axis("off")
    plt.show()
