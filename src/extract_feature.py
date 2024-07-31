import os
import pandas as pd
from tqdm import tqdm
import json
from openai import OpenAI
import src.utils as utils
import src.direct_ask as da

FEATURE_PROMPT = utils.PROMPT_feature

api_key = utils.api_key

def extract_features(city_year_country_list, output_file=utils.features_file):
    # da.get_response(city_year_country_list, utils.feature_json_file, PROMPT=FEATURE_PROMPT)``
    da.combine_files(utils.processed_data_file, utils.feature_json_file, output_file)
    
    print(f"Extracted features written to: {output_file}")
