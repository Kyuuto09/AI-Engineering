import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# 1. Load the data
data_file = os.path.join("../..", "..", "data", "internship_candidates_final_numeric.csv")
df = pd.read_csv(data_file)

# 2. Features (X) and Target (y)
X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ML Pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 7. VISUALIZATION (Probability Graph)
# ==========================================
print("\nGenerating probability graph...")

# grid of pixels representing all possible combinations of Experience and Test Scores.
exp_min = X['Experience'].min() - 0.5
exp_max = X['Experience'].max() + 0.5

test_min = X['EntryTestScore'].min() - 50
test_max = X['EntryTestScore'].max() + 5

# 100x100 points grid
xx_exp, yy_test = np.meshgrid(
    np.linspace(exp_min, exp_max, 100),
    np.linspace(test_min, test_max, 100)
)

# give the model all 5 columns to make a prediction.
# the grid for Experience and TestScore, and the AVERAGE for the others.
grid_data = pd.DataFrame({
    'Experience': xx_exp.ravel(),
    'Grade': [X['Grade'].mean()] * len(xx_exp.ravel()),
    'EnglishLevel': [X['EnglishLevel'].mean()] * len(xx_exp.ravel()),
    'Age': [X['Age'].mean()] * len(xx_exp.ravel()),
    'EntryTestScore': yy_test.ravel()
})

# Predict the PROBABILITY of being accepted (Class 1)
# predict_proba returns [prob_rejected, prob_accepted]. We want the second column ([:, 1]).
probs = model.predict_proba(grid_data)[:, 1]

# Reshape the long list of probabilities back into our 100x100 grid shape
probs = probs.reshape(xx_exp.shape)

# Draw the contour map (Heatmap)
plt.figure(figsize=(10, 6))
# cmap='RdYlGn' means Red (0%) to Yellow (50%) to Green (100%)
contour = plt.contourf(xx_exp, yy_test, probs, alpha=0.8, cmap='RdYlGn', levels=20)
plt.colorbar(contour, label='Probability of Acceptance')

# data points
plt.scatter(X['Experience'], X['EntryTestScore'], c=y, cmap='RdYlGn', edgecolors='black', s=50)

plt.xlabel('Experience (Years)')
plt.ylabel('Entry Test Score')
plt.title('Probability of SoftServe Internship Acceptance\n(Grade, Age, English held at average)')
plt.grid(True, alpha=0.3)
plt.savefig("softserve_internship_acceptance")
plt.show()