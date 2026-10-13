import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def preprocess_data(data):
    """
    Preprocesses the raw data from the JSON file.

    Args:
        data (dict): The raw data loaded from the JSON file.

    Returns:
        pd.DataFrame: A preprocessed DataFrame.
    """

    records = []
    for product_id, product_data in data.items():
        if "LTrecords" in product_data:
            for record_id, record in product_data["LTrecords"].items():
                record_copy = record.copy()

                # Exclude records with missing or invalid LT values
                lt_value = record_copy.get("LT")
                if lt_value is None or (isinstance(lt_value, str) and (lt_value.startswith("-") or lt_value == '')):
                    continue

                # Handle numeric fields to convert to float, mapping empty strings and non-numeric values to NaN
                numeric_fields = ["LS", "LT", "niu_months", "in_year"]
                for field in numeric_fields:
                    value = record_copy.get(field)
                    if isinstance(value, str):
                        try:
                            # Handle negative string values for LS
                            if field == "LS" and value.startswith("-"):
                                record_copy[field] = abs(float(value))
                            else:
                                record_copy[field] = float(value)
                        except (ValueError, TypeError):
                            record_copy[field] = np.nan
                    elif value is None:
                        record_copy[field] = np.nan

                records.append(record_copy)

    df = pd.DataFrame(records)

    # Drop rows with NaN values in critical columns for prediction
    df = df.dropna(subset=['LT', 'LS', 'in_year'])

    # Handle missing categorical values by filling with 'Unknown'
    categorical_cols = ['brand', 'model', 'chng_user', 'country', 'fail_rep', 'LTphase', 'in_condition',
                        'out_condition', 'out_method', 'price_range']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace('', 'Unknown').fillna('Unknown')

    # Convert remaining numerical columns to numeric type
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Feature Engineering
    # Calculate age at report date to create a new feature that could be a strong predictor
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['age_at_report_years'] = (df['report_date'].dt.year - df['in_year']) + (df['report_date'].dt.month / 12)

    # Drop columns that are no longer needed or are not features
    df = df.drop(columns=['ID_survey', 'ID_worker', 'report_date', 'in_month'])

    return df


def train_tuned_model(X, y):
    """
    Trains a tuned RandomForestRegressor model using GridSearchCV.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data (product lifetime).

    Returns:
        tuple: A tuple containing the best trained model and the feature list.
    """

    # Identify all categorical columns automatically
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert categorical features to one-hot encoded numerical format
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )

    # Fit the grid search to find the best parameters
    grid_search.fit(X_encoded, y)

    print("\nBest hyperparameters found by GridSearchCV:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Evaluate the best model
    best_model.fit(X_encoded, y)
    y_pred = best_model.predict(X_encoded)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print(f"\nFinal Model Performance on the entire dataset:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    return best_model, X_encoded.columns.tolist()


def predict_lifetime(model, new_data, model_features):
    """
    Predicts the lifetime for new data.

    Args:
        model: The trained machine learning model.
        new_data (dict): A dictionary of new data for a single product.
        model_features (list): The list of features the model was trained on.

    Returns:
        float: The predicted lifetime.
    """

    # Create a DataFrame from the new data
    new_df = pd.DataFrame([new_data])

    # Define categorical columns to be encoded
    categorical_cols = ['brand', 'model', 'chng_user', 'country', 'fail_rep', 'LTphase', 'in_condition',
                        'out_condition', 'out_method', 'price_range']

    # Apply the same preprocessing steps as the training data
    new_df['report_date'] = pd.to_datetime('2025-01-01')
    new_df['in_year'] = pd.to_numeric(new_df['in_year'], errors='coerce')
    new_df['age_at_report_years'] = (new_df['report_date'].dt.year - new_df['in_year']) + (
                new_df['report_date'].dt.month / 12)

    # Convert categorical features to one-hot encoded numerical format
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

    # Reindex to ensure all columns from the training set are present and in the correct order
    new_df_aligned = new_df_encoded.reindex(columns=model_features, fill_value=0)

    # Make prediction
    prediction = model.predict(new_df_aligned)
    return prediction[0]


if __name__ == '__main__':
    # Load the JSON file
    file_path = r"C:\Users\hp\Downloads\Consumer survey product lifetimes and durability of electronics\Consumer survey product lifetimes and durability of electronics\lt-platform-reports-export-10-26-2022-clean3-UPTOALL.json"
    with open(file_path, 'r') as f:
        full_data = json.load(f)

    # Preprocess the data
    preprocessed_df = preprocess_data(full_data)

    # Separate features and target variable
    target = 'LT'
    X = preprocessed_df.drop(columns=[target])
    y = preprocessed_df[target]

    # Train the tuned model on the entire dataset
    final_model, final_model_features = train_tuned_model(X, y)

    # Example new data for prediction (hypothetical)
    new_product_data = {
        'LS': '5.00',
        'niu_months': '12',
        'in_condition': '1',
        'in_month': '6',
        'in_year': '2018',
        'out_condition': '2',
        'out_method': '3',
        'price_range': '2',
        'brand': 'DELL',
        'model': 'Unknown models',
        'chng_user': 'No',
        'country': 'United States',
        'fail_rep': 'const',
        'LTphase': 'LT1',
    }

    # Predict lifetime for the new product
    predicted_lifetime = predict_lifetime(final_model, new_product_data, final_model_features)
    print("\nPredicted Lifetime for new product:", f"{predicted_lifetime:.2f} years")