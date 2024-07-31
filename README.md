Run the command in the root directory of the project, view more options in run.py. 

```
python -u "run.py" --tasks all
```

Original Data format:

```
city, year, country, target
```

Folder structure:

```
.
├── data
│   ├── co2
│   │   ├── processed
│   │   │   └── processed_data.csv
│   │   └── raw
│   │       └── data.csv
│   └── energy
│       ├── processed
│       │   └── processed_data.csv
│       └── raw
│           └── energy_consumption.csv
├── model
│   └── LLM-Research
│       └── Meta-Llama-3___1-8B-Instruct
├── README.md
├── res
│   ├── co2
│   │   ├── direct_ask_predictions.csv
│   │   ├── hidden_state_location.csv
│   │   ├── hidden_state_target.csv
│   │   └── tmp
│   │       ├── da_json_file.json
│   │       └── feature_json_file.json
│   └── energy
│       ├── direct_ask_predictions.csv
│       ├── extracted_features.csv
│       ├── hidden_state_location.csv
│       ├── hidden_state_target.csv
│       ├── rmse.json
│       └── tmp
│           ├── da_json_file.json
│           └── feature_json_file.json
├── run.py
└── src
    ├── data_processing.py
    ├── direct_ask.py
    ├── extract_feature.py
    ├── extract_hidden_state.py
    ├── ml.py
    └── utils.py

```
