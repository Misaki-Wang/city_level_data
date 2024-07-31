import argparse
import src.data_processing as dp
import src.direct_ask as da
import src.extract_feature as ef
import src.extract_hidden_state as ehs
import src.utils as utils
import src.ml as ml

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
    city_year_country_list = da.get_cities_years_country(utils.processed_data_file)
    
    if "direct_ask" in tasks:
        # Task 2: Directly ask the LLM for target prediction
        da.direct_ask_target_prediction(city_year_country_list, utils.direct_asking_file)

    if "extract_features" in tasks:
        # Task 3: Extract 4 specific features from LLM API
        ef.extract_features(city_year_country_list, utils.features_file)

    if "target" in tasks:
        # Task 4: Extract features from LLM hidden state
        ehs.extract_hidden_state_target(city_year_country_list, utils.hidden_state_target_file)
        
    if "location" in tasks:
        # Task 4: Extract features from LLM hidden state
        ehs.extract_hidden_state_location(city_year_country_list, utils.hidden_state_location_file)
        
    if "rmse" in tasks:
        ml.calculate_rmse_all()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks for data processing and LLM interaction.")
    parser.add_argument(
        '--tasks', 
        nargs='+', 
        choices=['process_data', 'direct_ask', 'extract_features', 'target', 'location', 'rmse'], 
        required=True, 
        help="Specify which tasks to execute. Choices are 'process_data', 'direct_ask', 'extract_features', 'target', 'location', 'rmse."
    )
    args = parser.parse_args()
    main(args.tasks)
