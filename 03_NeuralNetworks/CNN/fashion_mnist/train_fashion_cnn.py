import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# CNN model
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
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)

output_dir = os.path.dirname(__file__)
model_path = os.path.join(output_dir, "fashion_mnist_cnn.h5")
model.save(model_path)

saved_model = load_model(model_path)
test_loss, test_acc = saved_model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy (saved model): {test_acc:.4f}")

# Plot accuracy and loss
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(acc, label="Train accuracy")
axes[0].plot(val_acc, label="Val accuracy")
axes[0].set_title("Accuracy over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(loss, label="Train loss")
axes[1].plot(val_loss, label="Val loss")
axes[1].set_title("Loss over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()

plot_path = os.path.join(output_dir, "fashion_mnist_training.png")
plt.savefig(plot_path)
plt.show()
