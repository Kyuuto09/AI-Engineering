import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


def predict_electricity_consumption():
    # --- Data  ---
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_file = os.path.join(current_dir, "../..", "..", "data", "energy_usage.csv")
    df = pd.read_csv(data_file)

    # --- 2. Features (X) & Target (y) ---
    X = df[['temperature', 'humidity', 'hour', 'is_weekend']]
    y = df['consumption']

    # --- 3. Train/Test (80% Train, 20% Test) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Build and Train the Model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- 5. Make Predictions ---
    predicted_consumption = model.predict(X_test)

    # --- 6. Calculate the Error (%) ---
    error_percentage = mean_absolute_percentage_error(y_test, predicted_consumption) * 100
    print(f"Electricity Model Error Percentage: {error_percentage:.2f}%")

    # --- 7. Plot Actual vs. Predicted Consumption ---
    plt.figure(figsize=(8, 6))

    # energy/electricity
    plt.scatter(y_test, predicted_consumption, color='green', alpha=0.6, label='Predicted vs Actual')

    # "Perfect Prediction"
    max_val = max(y_test.max(), predicted_consumption.max())
    min_val = min(y_test.min(), predicted_consumption.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    # Formatting
    plt.title("Actual vs. Predicted Electricity Consumption")
    plt.xlabel("Actual Consumption (kWh)")
    plt.ylabel("Predicted Consumption (kWh)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.savefig("actual_vs_predicted_consumption.png")
    plt.show()


if __name__ == "__main__":
    predict_electricity_consumption()