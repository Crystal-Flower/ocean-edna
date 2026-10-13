import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def train_and_evaluate_model(file_path):
    """
    Trains and evaluates a RandomForestRegressor model to predict SALINITY.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        # Load the dataset. We use `encoding='latin1'` to fix the previous error,
        # `on_bad_lines='skip'` to skip any rows with formatting issues,
        # and `engine='python'` for more robust parsing of complex CSV files.
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')

        # ----------------------------------------
        # Data Preprocessing and Feature Engineering
        # ----------------------------------------

        # Drop columns with mostly missing or irrelevant data.
        # Irrelevant columns include identifiers and text comments.
        # We'll also drop Date and Time as they are difficult to use directly
        # and would require more complex feature engineering for this small dataset.
        columns_to_drop = [
            'ID', 'ID_2', 'Station', 'Haul', 'STN', 'Date', 'Time_(WA)',
            'UWST_-_Start', 'USWT-_Finish', 'Comments', 'Unnamed: 27',
            '*used_2%_bleach_-_0.4l_in_20L_water;_bleach_was_12.5%_which_was_not_diluted_further_2%_-_presumed_citizen_sciences_would_follow_as_per_instructions_and_whaterever_the_concentration_of_bleach_is_ignore_it_and_just_make_sure_the_amount_of_bleach_in_the_bucket_was_2%',
            'Filter_paper_split_into_two_samples'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Drop rows with any missing values. This is a simple approach for this small dataset.
        df.dropna(inplace=True)

        # Separate the target variable (y) from the features (X)
        target_column = 'SALINITY_(PSU)'
        features = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle the categorical 'Water_Mass' column using one-hot encoding.
        # This converts the categorical data into a format that the model can understand.
        X = pd.get_dummies(features, columns=['Water_Mass'], drop_first=True)

        # ----------------------------------------
        # Model Training and Evaluation
        # ----------------------------------------

        # Split the data into training and testing sets.
        # 80% for training and 20% for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the RandomForestRegressor model.
        # The RandomForest model is a good choice for regression tasks and is robust to noise.
        print("Training the RandomForestRegressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set.
        y_pred = model.predict(X_test)

        # Evaluate the model's performance.
        # R-squared (R2) score: Measures how much of the variance in the target variable
        # is explained by the model's features. A score closer to 1.0 is better.
        r2 = r2_score(y_test, y_pred)
        # Mean Absolute Error (MAE): The average absolute difference between the predicted
        # values and the actual values. A lower MAE is better.
        mae = mean_absolute_error(y_test, y_pred)

        print("\n--- Model Evaluation ---")
        print(f"R-squared Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Display the predicted vs actual values for a few test samples
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        print("\nSample of Actual vs Predicted values:")
        print(results_df.head().to_markdown(numalign="left", stralign="left"))

        return model

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    train_and_evaluate_model('eDNA.csv')
