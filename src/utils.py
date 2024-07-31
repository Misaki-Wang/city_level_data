# origin data columns
origin_city_columns = 'City name' # to be changed
origin_year_columns = 'Year of emission' # to be changed
origin_country_columns = 'Country' # to be changed
origin_target_columns = 'Total emissions (CDP) [tCO2-eq]' # to be changed

# File paths
task_folder = 'co2' # to be changed
city_column = 'city'
year_column = 'year'
country_column = 'country'
target_column = 'CO2_emission' # to be changed
prediction_column = f'{target_column}_pred'

features_selected = ['population', 'GDP', 'average annual temperature', 'built-up area']
features_hidden_state = [f"Feature{i}" for i in range(1, 33)]

# res file
input_file = f'data/{task_folder}/raw/data.csv'   # to be changed
processed_data_file = f'data/{task_folder}/processed/processed_data.csv'
direct_asking_file = f'res/{task_folder}/direct_ask_predictions.csv'
features_file = f'res/{task_folder}/extracted_features.csv'
hidden_state_target_file = f'res/{task_folder}/hidden_state_target.csv'
hidden_state_location_file = f'res/{task_folder}/hidden_state_location.csv'
rmse_file = f'res/{task_folder}/rmse.json'

# tmp file
da_json_file = f'res/{task_folder}/tmp/da_json_file.json'
da_json_clean_file = f'res/{task_folder}/tmp/da_json_clean_file.json'

feature_json_file = f'res/{task_folder}/tmp/feature_json_file.json'

# latten state
model_file = 'model/LLM-Research/Meta-Llama-3___1-8B-Instruct'

# PROMPT
# to be changed
PROMPT_da = """
Your task is to predict the CO2 emission of a city for a given year.
You will be provided with the city name, the year, and its country.

Organize your answer in a JSON object containing the following keys:
- city: The name of the city you are provided
- year: The year you are provided
- country: The country of the city you are provided
- CO2_emissions: The predicted CO2 emissions value for the city in the given year, expressed in metric tons of CO2 equivalent

The data is as follows:
- city: {city}
- year: {year}
- country: {country}
"""

# to be changed
PROMPT_target = """
Provide detailed information on factors that could affect CO2 emissions in the city for the given year:
- City: {city}
- Year: {year}
- Country: {country}
"""

PROMPT_feature = """
You will be provided with the city name, the year, and its country.

Organize your answer in a JSON object containing the following keys:
- city: The name of the city you are provided
- year: The year you are provided
- country: The country of the city you are provided
- population: The population of the city, scaled from 0 to 10.0
- GDP: The Gross Domestic Product of the city, scaled from 0 to 10.0
- average annual temperature: The average annual temperature of the city, scaled from 0 to 10.0
- built-up area: The built-up area of the city, scaled from 0 to 10.0

The data is as follows:
- city: {city}
- year: {year}
- country: {country}
"""

PROMPT_location = """
Provide detailed information about the city for the given year:
- City: {city}
- Year: {year}
- Country: {country}
"""

# Configuration parameters
api_key = "sk-lmMVrXFrHjz21ZcR9f2b8f3eB82d4cAf85B066D6C91bB9Dd"
model = "gpt-4o"
temperature = 0.01
random_seed = 42
