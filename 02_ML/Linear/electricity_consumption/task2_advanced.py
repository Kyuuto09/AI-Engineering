import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn is good for improved styling (might use it in the future as well !!!)
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


def predict_electricity_with_categories():
    # --- Data  ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "../..", "..", "data", "energy_usage_plus.csv")
    df = pd.read_csv(data_file)

    # --- 2. Features (X) & Target (y) ---
    X = df[['temperature', 'humidity', 'hour', 'is_weekend', 'season', 'district_type']]
    y = df['consumption']

    # --- 3. Train/Test (80% Train, 20% Test) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Build and Train the Model (The AI Engineering Way) ---
    # We tell the preprocessor to use OneHotEncoder ONLY on the text columns.
    # drop='first' prevents a math issue called "dummy variable trap"
    # remainder='passthrough' tells it to leave our numbers (temp, hour) alone.
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='first'), ['season', 'district_type']),
        remainder='passthrough'
    )

    # Bundle the preprocessor and the regression model together into a Pipeline
    model = make_pipeline(
        preprocessor,
        LinearRegression()
    )

    # Train the pipeline (it will automatically encode the text, then train the model)
    model.fit(X_train, y_train)

    # --- 5. Make Predictions ---
    # The pipeline automatically encodes the test data too! No data leakage.
    predicted_consumption = model.predict(X_test)

    # --- 6. Calculate the Error (%) ---
    error_percentage = mean_absolute_percentage_error(y_test, predicted_consumption) * 100
    print(f"Advanced Electricity Model Error Percentage: {error_percentage:.2f}%")

    # --- 7. Plot Actual vs. Predicted Consumption (Using Seaborn) ---
    # ! NOTE: Seaborn makes the plot look much more professional out of the box
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # ! NOTE: Using sns.scatterplot instead of plt.scatter (USEFUL, learn more later !!! )
    sns.scatterplot(x=y_test, y=predicted_consumption, color='green', alpha=0.6, label='Predicted vs Actual')

    # Line
    max_val = max(y_test.max(), predicted_consumption.max())
    min_val = min(y_test.min(), predicted_consumption.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    plt.title("Actual vs. Predicted Electricity Consumption\n(Including Season & District)")
    plt.xlabel("Actual Consumption (kWh)")
    plt.ylabel("Predicted Consumption (kWh)")
    plt.legend()

    plt.savefig("advanced_actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    predict_electricity_with_categories()