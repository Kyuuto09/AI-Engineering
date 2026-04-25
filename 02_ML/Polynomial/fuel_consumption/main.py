import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline

#  ---------
#  DATA Load
#  ---------
data_file = os.path.join("../..", "..", "data", "fuel_consumption_vs_speed.csv")
df = pd.read_csv(data_file)


X = df[['speed_kmh']]
y = df['fuel_consumption_l_per_100km']

# 2. DATA Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_degree = 1
best_mse = float('inf')
metrics_history = {}

print("--- Evaluating Polynomial Degrees ---")

# 3. Find optimal polynomial degree from 1 to 5
for degree in range(1, 6):
    # a pipeline: Scale -> Polynomial Features -> Linear Regression
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # Model training
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics_history[degree] = {'MSE': mse, 'MAE': mae}
    print(f"Degree {degree} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    # Update the best degree based on MSE
    if mse < best_mse:
        best_mse = mse
        best_degree = degree

print(f"\nOptimal degree chosen: {best_degree}")

# 4. Train the final model using the BEST degree on ALL available data
final_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=best_degree),
    LinearRegression()
)
final_model.fit(X, y)  # Notice we fit on X, y (the full dataset)

# 5. Predict fuel consumption for specific speeds: 35, 95, 140
speeds_to_predict = pd.DataFrame({'speed_kmh': [35, 95, 140]})
predictions = final_model.predict(speeds_to_predict)

print("\n--- Final Predictions ---")
for speed, pred in zip(speeds_to_predict['speed_kmh'], predictions):
    print(f"Speed: {speed} km/h -> Predicted Fuel Consumption: {pred:.2f} L/100km")

# 6. Plot the results
X_plot = pd.DataFrame({'speed_kmh': np.linspace(X['speed_kmh'].min(), X['speed_kmh'].max(), 100)})
y_plot = final_model.predict(X_plot)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_plot, color='red', label=f'Polynomial Degree {best_degree}')
plt.scatter(speeds_to_predict, predictions, color='green', marker='X', s=100, label='New Predictions')
plt.title('Fuel Consumption vs Speed')
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.legend()
plt.grid(True)
plt.savefig("fuel_consumption")
plt.show()


# --- Extracting the formula for Desmos ---

# We train a temporary model WITHOUT the StandardScaler
# purely to get a clean mathematical formula for raw X values.
poly_desmos = PolynomialFeatures(degree=4)
X_poly_raw = poly_desmos.fit_transform(X)

desmos_model = LinearRegression()
desmos_model.fit(X_poly_raw, y)

# Extract the numbers (coefficients and intercept)
coefs = desmos_model.coef_
intercept = desmos_model.intercept_

print("\n--- Desmos Formula ---")
print("Copy and paste this exact line into Desmos:")

# We use .8f to capture very small decimals, otherwise x^4 might round to 0
formula = f"y = {intercept:.8f} + ({coefs[1]:.8f} * x) + ({coefs[2]:.8f} * x^2) + ({coefs[3]:.8f} * x^3) + ({coefs[4]:.8f} * x^4)"
print(formula)