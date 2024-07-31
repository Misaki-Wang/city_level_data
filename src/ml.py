import pandas as pd
import numpy as np
import src.utils as utils
from sklearn.model_selection import train_test_split, cross_val_predict
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import os

def calculate_rmse(df, target_column=utils.target_column, prediction_column=utils.prediction_column):
    # Ensure the target and prediction columns exist in the dataframe
    if target_column not in df.columns or prediction_column not in df.columns:
        raise ValueError(f"Columns '{target_column}' and/or '{prediction_column}' not found in the DataFrame.")
    # Calculate the squared differences
    squared_diff = (df[target_column] - df[prediction_column]) ** 2
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    # Calculate the root of the mean squared differences
    rmse = np.sqrt(mean_squared_diff)
    
    return rmse

def avg_prediction(processed_data_file = utils.processed_data_file, target_column=utils.target_column, prediction_column=utils.prediction_column):
    # Read the processed data file into a DataFrame
    df = pd.read_csv(processed_data_file, encoding='utf-8')
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the DataFrame.")
    # Calculate the average of the target column
    average_value = df[target_column].mean()
    # Add the prediction column with all values set to the average of the target column
    df[prediction_column] = average_value
    return df

# machine learning task
def perform_ml_task_and_add_predictions(data_file, features, target_column='energy_consumption', prediction_column='energy_consumption_pred'):
    # Read the processed data file into a DataFrame
    df = pd.read_csv(data_file, encoding='utf-8')
    
    # Check if the required columns exist in the DataFrame
    missing_columns = [col for col in features + [target_column] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")
    
    # Prepare the feature matrix (X) and target vector (y)
    X = df[features]
    y = df[target_column]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Define the model
    model = XGBRegressor(n_estimators=100, random_state=42)
    
    # Perform cross-validation and get predictions
    predictions = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
    
    # Add the prediction column to the DataFrame
    df[prediction_column] = predictions
    
    return df

# calculate 5 tasks separately
def calculate_direct_ask_rmse(direct_asking_file = utils.direct_asking_file, target_column=utils.target_column, prediction_column=utils.prediction_column):
    # Read the direct asking file into a DataFrame
    df_direct = pd.read_csv(direct_asking_file, encoding='utf-8')
    # Calculate the RMSE
    rmse = calculate_rmse(df_direct, target_column, prediction_column)
    return rmse

def calculate_feature_rmse():
    features = utils.features_selected
    df = perform_ml_task_and_add_predictions(utils.features_file, features)
    rmse = calculate_rmse(df)
    return rmse

def calculate_avg_rmse(processed_data_file = utils.processed_data_file, target_column=utils.target_column, prediction_column=utils.prediction_column):
    df_avg = avg_prediction(processed_data_file, target_column, prediction_column)
    rmse = calculate_rmse(df_avg, target_column, prediction_column)
    return rmse

def calculate_target_rmse():
    features = utils.features_hidden_state
    df = perform_ml_task_and_add_predictions(utils.hidden_state_target_file, features)
    rmse = calculate_rmse(df)
    return rmse

def calculate_location_rmse():
    features = utils.features_hidden_state
    df = perform_ml_task_and_add_predictions(utils.hidden_state_location_file, features)
    rmse = calculate_rmse(df)
    return rmse

# calculate_rmse_all
def calculate_rmse_all(output_file=utils.rmse_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    rmse_avg = calculate_avg_rmse()
    rmse_direct_ask =  calculate_direct_ask_rmse()
    rmse_feature = calculate_feature_rmse()
    rmse_target = calculate_target_rmse()
    rmse_location = calculate_location_rmse()
    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'Method': ['Features', 'Target', 'Location', 'Direct Ask', 'AvgY'],
        'RMSE': [rmse_feature, rmse_target, rmse_location, rmse_direct_ask, rmse_avg]
    })
    # Save the results to a CSV file
    results.to_json(output_file, orient='records', lines=True, indent=4)
    
    print(f"RMSE results saved to {output_file}")
