import pandas as pd
import json
import os
from tqdm import tqdm
from openai import OpenAI
import src.utils as utils


PROMPT = utils.PROMPT_da
api_key = utils.api_key

def get_cities_years_country(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    city_year_country_list = df[['city', 'year', 'country']].drop_duplicates().values.tolist()
    return city_year_country_list

def get_complate_openai(text):
    client = OpenAI(api_key=api_key, base_url="https://api.zyai.online/v1")
    response = client.chat.completions.create(
        model=utils.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text},
        ],
        stream=False,
        temperature=utils.temperature,
    )
    answer = response.choices[0].message.content
    return answer

def extract_json_objects(text):
    objects = []
    depth = 0
    start = None

    for i, char in enumerate(text):
        if (char == '{'):
            if depth == 0:
                start = i
            depth += 1
        elif (char == '}'):
            depth -= 1
            if (depth == 0 and start is not None):
                objects.append(text[start:i+1])

    parsed_json_objects = []
    for obj in objects:
        try:
            parsed_json_objects.append(json.loads(obj))
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} for object: {obj}")

    return parsed_json_objects

def get_response(city_year_country_list, output_file, PROMPT = PROMPT):
    if os.path.exists(output_file):
        os.remove(output_file)
    for city, year, country in tqdm(city_year_country_list, desc="Direct Asking", leave=True):
        try:
            prompt_text = PROMPT.format(city=city, year=year, country=country)
            response = get_complate_openai(prompt_text)
            if response is None:
                continue
            js = extract_json_objects(response)
        except Exception as e:
            print(f"Error during prediction or extraction: {e}")
            continue
    
        try:
            with open(output_file, 'a', encoding='utf-8') as file:
                for entry in js:
                    json.dump(entry, file, ensure_ascii=False)
                    file.write('\n')
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error writing to file: {e}")
            
    print(f"Response from API written to: {output_file}")

def clean_data(file_path, output_file, target_name = utils.target_column, prediction_name = utils.prediction_column):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_lines = [json.loads(line) for line in f]
    predictions_df = pd.DataFrame(json_lines)
    
    predictions_df.rename(columns={target_name: prediction_name}, inplace=True)
    
    predictions_df = predictions_df[predictions_df[prediction_name].apply(pd.to_numeric, errors='coerce').notna()]
    
    predictions_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"Cleaned data written to: {output_file}")

def combine_files(original_file, direct_asking_file, output_file, city_column=utils.city_column, year_column=utils.year_column, country_column=utils.country_column):
    try:
        original_df = pd.read_csv(original_file, encoding='utf-8')
        
        with open(direct_asking_file, 'r', encoding='utf-8') as f:
            json_lines = [json.loads(line) for line in f]
        predictions_df = pd.DataFrame(json_lines)
        
        combined_df = pd.merge(original_df, predictions_df, left_on=[city_column, year_column, country_column], right_on=[city_column, year_column, country_column], how='left')
        
        # combined_df.drop(columns=[city_column, year_column, country_column], inplace=True)
        
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Combined files written to: {output_file}")
    except Exception as e:
        print(f"Error combining files with pandas: {e}")

def delete_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def direct_ask_target_prediction(city_year_country_list, output_file=utils.direct_asking_file):
    get_response(city_year_country_list, utils.da_json_file, PROMPT)
    clean_data(utils.da_json_file, utils.da_json_clean_file)
    combine_files(utils.processed_data_file, utils.da_json_clean_file, output_file)
    
    delete_files(utils.da_json_clean_file)
    
    print("End of direct ask target prediction")
