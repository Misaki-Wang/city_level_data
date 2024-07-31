import pandas as pd
import os
import src.utils as utils

def process_data(input_file, output_file, 
                 origin_city_columns=utils.origin_city_columns, origin_year_columns=utils.origin_year_columns,
                 origin_country_columns=utils.origin_country_columns, origin_target_columns=utils.origin_target_columns,
                 city_column=utils.city_column, country_column=utils.country_column,
                 year_column=utils.year_column, target_column=utils.target_column):
    df = pd.read_csv(input_file, encoding='utf-8')
    # Check if the required original columns exist in the DataFrame
    missing_columns = [col for col in [origin_city_columns, origin_country_columns, origin_year_columns, origin_target_columns] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")
    
    # Select and rename the columns
    df = df[[origin_city_columns, origin_country_columns, origin_year_columns, origin_target_columns]]
    df.columns = [city_column, country_column, year_column, target_column]
    
    # Remove leading and trailing spaces from 'city' column
    df[city_column] = df[city_column].str.strip()
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the processed data to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Processed data written to: {output_file}")
