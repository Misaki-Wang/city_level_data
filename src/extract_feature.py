import pandas as pd
from tqdm import tqdm
import json
from openai import OpenAI
import src.utils as utils

FEATURE_PROMPT = """
You will be provided with the city name, the year, and its country.

Organize your answer in a JSON object containing the following keys:
- city: The name of the city you are provided
- year: The year you are provided
- country: The country of the city you are provided
- feature_1: Feature 1 scaled from 0 to 10
- feature_2: Feature 2 scaled from 0 to 10
- feature_3: Feature 3 scaled from 0 to 10
- feature_4: Feature 4 scaled from 0 to 10

The data is as follows:
- city: {city}
- year: {year}
- country: {country}
"""

api_key = utils.api_key

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

def extract_features(city_year_country_list, output_file):
    results = []
    for city, year, country in tqdm(city_year_country_list, desc="Feature Extraction", leave=True):
        try:
            prompt_text = FEATURE_PROMPT.format(city=city, year=year, country=country)
            response = get_complate_openai(prompt_text)
            if response is None:
                continue
            js = extract_json_objects(response)
            for entry in js:
                results.append(entry)
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            continue

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Extracted features written to: {output_file}")
