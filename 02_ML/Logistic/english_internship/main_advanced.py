import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline


def predict_internship_with_text_english():
    #current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load the data
    data_file = os.path.join("../..", "..", "data", "internship_candidates_cefr_final.csv")
    df = pd.read_csv(data_file)

    # Note: Ensure your CSV has EnglishLevel as text (e.g., 'B1', 'B2', 'C1')

    # 2. Separate Features (X) and Target (y)
    X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
    y = df['Accepted']

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build the Preprocessor
    # We apply OneHotEncoder ONLY to the 'EnglishLevel' column.
    # We apply StandardScaler to the numerical columns.
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='first'), ['EnglishLevel']),
        (StandardScaler(), ['Experience', 'Grade', 'Age', 'EntryTestScore'])
    )

    # 5. Build the Pipeline
    model = make_pipeline(
        preprocessor,
        LogisticRegression()
    )

    # 6. Train the Model
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("--- Advanced Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # ==========================================
    # 8. VISUALIZATION (Probability)
    # ==========================================
    print("\nGenerating probability graph...")
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the boundaries with padding so dots don't get cut off
    exp_min = X['Experience'].min() - 0.5
    exp_max = X['Experience'].max() + 0.5
    test_min = X['EntryTestScore'].min() - 50
    test_max = X['EntryTestScore'].max() + 50


    xx_exp, yy_test = np.meshgrid(
        np.linspace(exp_min, exp_max, 100),
        np.linspace(test_min, test_max, 100)
    )

    # Best Practice: Find the most common text value (Mode) for categorical data
    most_common_english = X['EnglishLevel'].mode()[0]
    average_grade = X['Grade'].mean()
    average_age = X['Age'].mean()

    # Create the DataFrame for our grid ("Phantom Student") NOTE !!!
    grid_data = pd.DataFrame({
        'Experience': xx_exp.ravel(),
        'Grade': [average_grade] * len(xx_exp.ravel()),
        'EnglishLevel': [most_common_english] * len(xx_exp.ravel()),  # Frozen as text!
        'Age': [average_age] * len(xx_exp.ravel()),
        'EntryTestScore': yy_test.ravel()
    })

    # Predict the PROBABILITY of being accepted (Class 1)
    probs = model.predict_proba(grid_data)[:, 1]
    probs = probs.reshape(xx_exp.shape)

    # Draw the contour map (Heatmap)
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(xx_exp, yy_test, probs, alpha=0.8, cmap='RdYlGn', levels=20)
    plt.colorbar(contour, label='Probability of Acceptance')

    # Plot the actual data points over the map
    plt.scatter(X['Experience'], X['EntryTestScore'], c=y, cmap='RdYlGn', edgecolors='black', s=50)

    # Formatting
    plt.xlabel('Experience (Years)')
    plt.ylabel('Entry Test Score')
    plt.title(f'Probability of SoftServe Acceptance\n(Grade & Age = Avg | English = {most_common_english})')
    plt.grid(True, alpha=0.3)

    plt.savefig("advanced_logistic_heatmap.png")
    plt.show()


if __name__ == "__main__":
    predict_internship_with_text_english()