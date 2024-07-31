# Configuration parameters
api_key = "sk-lmMVrXFrHjz21ZcR9f2b8f3eB82d4cAf85B066D6C91bB9Dd"
model = "gpt-4o"
temperature = 0.01

# File paths
task_folder = 'energy'
city_column = 'city'
year_column = 'year'
country_column = 'country'
target_column = 'energy_consumption'
prediction_column = 'energy_consumption_pred'

# res file
input_file = f'data/{task_folder}/raw/energy_consumption.csv'
processed_data_file = f'data/{task_folder}/processed/processed_data.csv'
direct_asking_file = f'res/{task_folder}/direct_ask_predictions.csv'
features_file = f'res/{task_folder}/extracted_features.csv'
hidden_state_features_file = f'res/{task_folder}/hidden_state_features.csv'

# tmp file
da_json_file = f'res/{task_folder}/tmp/da_json_file.json'
da_json_clean_file = f'res/{task_folder}/tmp/da_json_clean_file.json'

PROMPT_da = """
Your task is to predict the energy consumption of a city for a given year.
You will be provided with the city name, the year, and its country.

Organize your answer in a JSON object containing the following keys:
- city: The name of the city you are provided
- year: The year you are provided
- country: The country of the city you are provided
- energy_consumption: Final energy consumption (10000 tons of standard coal) in the city for the given year

The data is as follows:
- city: {city}
- year: {year}
- country: {country}
"""