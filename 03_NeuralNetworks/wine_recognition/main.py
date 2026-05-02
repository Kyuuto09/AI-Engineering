import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  # <-- FIX 1: Added missing import!
import numpy as np

# FIX 2: Set the random seed so the AI doesn't get a "Bad Brain" starting point
tf.keras.utils.set_random_seed(42)


def classify_wine():
    # Wine data set
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    print("Feature sample (raw):", X[0][:4], "...")

    # --- Normalize features ---
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Feature sample (Normalized):", X[0][:4], "...")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = tf.keras.Sequential(
        [
            # 13 features in dataset
            layers.Dense(16, activation="relu", input_shape=(13,)),
            layers.Dense(8, activation="relu"),
            # Output layer: 3 neurons because 3 classes of wine
            layers.Dense(3),
        ]
    )

    # --- Compile ---
    model.compile(
        optimizer="adam",
        # from_logits=True is highly recommended in TF for numerical stability
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()

    # --- Train ---
    print("\n--- Starting Training ---")
    # 100 epochs to give the AI enough time to learn
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

    # --- Evaluate ---
    print("\n--- Evaluating on Test Data ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # --- Classification Report ---
    all_pred_logits = model.predict(X_test, verbose=0)
    all_pred_classes = np.argmax(all_pred_logits, axis=1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, all_pred_classes, target_names=wine_data.target_names
        )
    )

    # --- Prediction ---
    print("\n--- Multiple Random Predictions ---")

    # FIX 3: Randomize the guesses!
    random_indices = np.random.choice(len(X_test), size=5, replace=False)

    for i, random_index in enumerate(random_indices):
        sample = X_test[random_index].reshape(1, -1)
        actual_class = y_test[random_index]

        # Predict
        pred_logits = model.predict(sample, verbose=0)
        pred_class = np.argmax(pred_logits, axis=1)[0]  # argmax finds index

        print(
            f"\nRandom Wine {i+1} (Index {random_index}) Raw Logits: {pred_logits[0]}"
        )
        print(
            f"Wine {i+1}: Predicted -> {wine_data.target_names[pred_class]} | Actual -> {wine_data.target_names[actual_class]}"
        )


if __name__ == "__main__":
    classify_wine()
