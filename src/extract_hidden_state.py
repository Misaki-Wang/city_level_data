from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import pearsonr
import torch.nn as nn
from tqdm import tqdm
import src.utils as utils
import os

PROMPT_target = utils.PROMPT_target
PROMPT_location = utils.PROMPT_location
model_path = utils.model_file

def concat_mean_max_pooling(hidden_states):
    mean_pooled = hidden_states.mean(dim=1)
    max_pooled = hidden_states.max(dim=1).values
    pooled = torch.cat((mean_pooled, max_pooled), dim=-1)
    return pooled

class DimensionalityReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -0.1, 0.1)
    
    def forward(self, x):
        return self.linear(x)

dim_reducer = DimensionalityReducer(8192, 32).to('cpu')  

def reduce_dimensionality(features):
    features = features.to(torch.float32).to('cpu') 
    return dim_reducer(features)

def extract_features(hidden_states):
    hidden_states = hidden_states.to('cpu') 
    pooled_features = concat_mean_max_pooling(hidden_states)
    reduced_features = reduce_dimensionality(pooled_features)
    return reduced_features

def get_last_hidden_state(prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
    response = model.model(**inputs, output_hidden_states=True).last_hidden_state
    return response

def get_features_test(prompt, model, tokenizer):
    res = get_last_hidden_state(prompt, model, tokenizer)
    return extract_features(res).tolist()[0]

def extract_hidden_state(city_year_country_list, output_file, PROMPT):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.float16)
    
    feature_list = []
    for city, year, country in tqdm(city_year_country_list, desc="Hidden state extraction", leave=True):
        prompt = PROMPT.format(city=city, year=year, country=country)
        feature = get_features_test(prompt, model, tokenizer)
        feature_list.append([city]+feature)
        
    feature_df = pd.DataFrame(feature_list, columns=[utils.city_column]+[f"Feature{i}" for i in range(1, 33)])
    data = pd.read_csv(utils.processed_data_file, encoding='utf-8')
    df = pd.merge(data, feature_df, on=utils.city_column)
    
    df.to_csv(output_file, index=False)
    
    print(f"Hidden state features extracted and saved to {output_file}")

def extract_hidden_state_target(city_year_country_list, output_file=utils.hidden_state_target_file, PROMPT=PROMPT_target):
    extract_hidden_state(city_year_country_list, output_file, PROMPT)
    
def extract_hidden_state_location(city_year_country_list, output_file=utils.hidden_state_location_file, PROMPT=utils.PROMPT_location):
    extract_hidden_state(city_year_country_list, output_file, PROMPT)