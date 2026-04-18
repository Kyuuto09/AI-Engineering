import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)


def linear_regression():
    # --- 1. Load the Data ---
    data_file = os.path.join("../..", "..", "data", "cars.csv")
    df = pd.read_csv(data_file)

    # --- 2. Features (X) & Target (y) ---
    X = df[['year', 'engine_volume', 'mileage', 'horsepower']]
    y = df['price']

    # --- 3. Train/Test Split (80%/20%) ---
    # random_state=42 just ensures we get the exact same random split every time we run the script.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Build and Train the Model ---
    # This is like instantiating an object in Java: LinearRegression model = new LinearRegression();
    model = LinearRegression()

    # fit calculates the math to draw the best possible line through the training data.
    model.fit(X_train, y_train)

    # --- 5. Make Predictions ---
    # X_test to find out the answers
    predicted_prices = model.predict(X_test)

    # --- 6. Calculate the Error (%) ---
    error_percentage = mean_absolute_percentage_error(y_test, predicted_prices) * 100
    print(f"------------\nModel Error Percentage: {error_percentage:.2f}%\n------------")

    # -- TEST
    print(f"\nPredicted Prices\n{predicted_prices}")

    # --- 7. Plot Actual vs. Predicted Prices ---
    plt.figure(figsize=(8, 6))

    # Plot the actual prices vs what the model guessed.
    # alpha=0.6 allows to see where dots overlap.
    plt.scatter(y_test, predicted_prices, color='blue', alpha=0.6, label='Predicted vs Actual')

    # Draw a "Perfect Prediction" line.
    # If the model guessed perfectly, every blue dot would land exactly on this red dashed line.
    max_price = max(y_test.max(), predicted_prices.max())
    min_price = min(y_test.min(), predicted_prices.min())
    plt.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--', label='Perfect Prediction')

    # Formatting
    plt.title("Actual vs. Predicted Car Prices")
    plt.xlabel("Actual Price (UAH)")
    plt.ylabel("Predicted Price (UAH)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.savefig("actual_vs_predicted_prices.png")
    plt.show()


if __name__ == "__main__":
    linear_regression()