import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io

# The provided file is accessible via a special file object.
# We'll use this object to read the data into a pandas DataFrame.
try:
    with open('Fish.csv', 'r') as file:
        data = file.read()
    df = pd.read_csv(io.StringIO(data))
except FileNotFoundError:
    print("Error: 'Fish.csv' not found. Please ensure the file is available.")
    exit()

# --- Data Preparation ---

# Identify features (X) and target (y)
# The 'Species' column is our target variable. All other columns are features.
X = df.drop('Species', axis=1)
y = df['Species']

# Since 'Species' is a categorical text column, we need to convert it to numerical labels.
# This is required for most machine learning algorithms.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features. This is a crucial step for many models to prevent features
# with larger numerical values from dominating the learning process.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Splitting the Data ---

# Split the dataset into training and testing sets.
# We'll use 80% of the data for training the model and 20% for testing its performance.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# --- Model Training ---

# Initialize the RandomForestClassifier. Random Forest is an ensemble learning method
# that can achieve high accuracy and is less prone to overfitting.
# The n_estimators parameter specifies the number of trees in the forest.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data.
print("Training the Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- Model Evaluation on Test Data ---

# Make predictions on the unseen test set to evaluate true model performance.
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the test set.
accuracy = accuracy_score(y_test, y_pred)

# Display the results.
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# Check if the accuracy meets the 98% threshold.
if accuracy >= 0.98:
    print("Success! The model has achieved an accuracy of 98% or higher.")
else:
    print("The model did not reach the 98% accuracy threshold. Consider tuning hyperparameters or trying a different model.")

# --- Optional: Making a prediction on new data ---

# Example of how you would use the trained model on a new data point.
# You need to provide all feature values in the same order and format.
new_fish_data = pd.DataFrame({
    'Weight': [500],
    'Length1': [26.8],
    'Length2': [29.7],
    'Length3': [34.5],
    'Height': [14.1795],
    'Width': [5.2785]
})

# Scale the new data using the same scaler object used for training.
new_fish_scaled = scaler.transform(new_fish_data)

# Predict the species. The prediction will be a numerical label.
prediction_label = model.predict(new_fish_scaled)

# Convert the numerical label back to the original species name.
predicted_species = label_encoder.inverse_transform(prediction_label)

print(f"\nPrediction for the new fish data point: {predicted_species[0]}")