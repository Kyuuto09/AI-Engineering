import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers

# Set random seed for a stable "brain"
tf.keras.utils.set_random_seed(42)


def predict_fuel_nn():
    # ---------
    # 1. DATA Load
    # ---------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(
        current_dir, "..", "..", "data", "fuel_consumption_vs_speed.csv"
    )

    # trip_time' and 'engine_type' columns
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Please check the path.")
        return

    # ---------
    # 2. Features & Target
    # ---------
    # Backfill optional columns if the CSV only contains speed + fuel consumption.
    if "trip_time" not in df.columns:
        df["trip_time"] = 0.0
        print("Warning: 'trip_time' missing in CSV. Using 0.0 for all rows.")
    if "engine_type" not in df.columns:
        df["engine_type"] = "unknown"
        print("Warning: 'engine_type' missing in CSV. Using 'unknown' for all rows.")

    X = df[["speed_kmh", "trip_time", "engine_type"]]
    y = df["fuel_consumption_l_per_100km"]

    # ---------
    # 3. Train/Test Split
    # ---------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------
    # 4. Preprocessing (The AI Engineering Way)
    # ---------
    # Handle the text column ('engine_type') and numerical columns separately
    preprocessor = make_column_transformer(
        (
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ),
            ["engine_type"],
        ),
        (StandardScaler(), ["speed_kmh", "trip_time"]),
    )

    # Transform the data BEFORE giving it to TensorFlow
    # TensorFlow expects pure numbers, so we run the preprocessor here
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Determine how many inputs the NN will receive after OneHotEncoding
    input_dim = X_train_processed.shape[1]

    # ---------
    # 5. Build the Neural Network
    # ---------
    model = tf.keras.Sequential(
        [
            layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            # Output layer: 1 neuron, NO activation function (linear output for regression)
            layers.Dense(1),
        ]
    )

    # Compile for Regression
    model.compile(
        optimizer="adam",
        loss="mse",  # Mean Squared Error
        metrics=["mae"],  # Mean Absolute Error
    )

    model.summary()

    # ---------
    # 6. Train the Model
    # ---------
    print("\n--- Training Neural Network ---")
    # Using validation_split lets us see if the model is overfitting during training
    history = model.fit(
        X_train_processed,
        y_train,
        epochs=150,
        batch_size=16,
        verbose=1,
        validation_split=0.1,
    )

    # ---------
    # 7. Evaluate and Compare
    # ---------
    print("\n--- Neural Network Evaluation ---")
    y_pred = model.predict(X_test_processed, verbose=0)

    nn_mse = mean_squared_error(y_test, y_pred)
    nn_mae = mean_absolute_error(y_test, y_pred)

    print(f"Neural Network MSE: {nn_mse:.4f}")
    print(f"Neural Network MAE: {nn_mae:.4f}")

    # ---------
    # 8. Predict New Specific Cases
    # ---------
    print("\n--- Final Predictions ---")

    # We must provide all 3 features for our new predictions
    new_data = pd.DataFrame(
        {
            "speed_kmh": [35, 95, 140],
            "trip_time": [20, 60, 120],  # Mock trip times
            "engine_type": ["gasoline", "diesel", "gasoline"],  # Mock engine types
        }
    )

    # Run new data through the exact same preprocessor!
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
        module="sklearn",
    )

    new_data_processed = preprocessor.transform(new_data)
    new_predictions = model.predict(new_data_processed, verbose=0)

    for i in range(len(new_data)):
        print(
            f"Speed: {new_data['speed_kmh'][i]} km/h | Time: {new_data['trip_time'][i]}m | Engine: {new_data['engine_type'][i]}"
        )
        print(
            f"   -> Predicted Fuel Consumption: {new_predictions[i][0]:.2f} L/100km\n"
        )


if __name__ == "__main__":
    predict_fuel_nn()
