import pandas as pd
import os

def process_data(input_file, output_file, city_column, country_column, year_column, target_column):
    df = pd.read_csv(input_file, encoding='utf-8')
    df = df[[city_column, country_column, year_column, target_column]]
    df["city"] = df["city"].str.strip()  # Remove leading and trailing spaces from 'city' column
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Processed data written to: {output_file}")
