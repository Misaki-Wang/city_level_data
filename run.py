import argparse
import src.data_processing as dp
import src.direct_ask as da
import src.extract_feature as ef
import src.extract_hidden_state as ehs
import src.utils as utils

def main(tasks):
    if "process_data" in tasks:
        # Task 1: Process the data
        dp.process_data(
            utils.input_file,
            utils.processed_data_file,
            utils.city_column,
            utils.country_column,
            utils.year_column,
            utils.target_column
        )
    
    if "direct_ask" in tasks:
        # Get unique city, year, and country combinations
        city_year_country_list = da.get_cities_years_country(utils.processed_data_file)
        # Task 2: Directly ask the LLM for target prediction
        da.direct_ask_target_prediction(city_year_country_list, utils.direct_asking_file)

    if "extract_features" in tasks:
        # Get unique city, year, and country combinations
        city_year_country_list = da.get_cities_years_country(utils.processed_data_file)
        # Task 3: Extract 4 specific features from LLM API
        ef.extract_features(city_year_country_list, utils.features_file)

    if "extract_hidden_state" in tasks:
        # Get unique city, year, and country combinations
        city_year_country_list = da.get_cities_years_country()
        # Task 4: Extract features from LLM hidden state
        ehs.extract_hidden_state_features(city_year_country_list, utils.hidden_state_features_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks for data processing and LLM interaction.")
    parser.add_argument(
        '--tasks', 
        nargs='+', 
        choices=['process_data', 'direct_ask', 'extract_features', 'extract_hidden_state'], 
        required=True, 
        help="Specify which tasks to execute. Choices are 'process_data', 'direct_ask', 'extract_features', 'extract_hidden_state'."
    )
    args = parser.parse_args()
    main(args.tasks)
