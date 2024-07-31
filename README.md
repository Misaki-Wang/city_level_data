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
│           ├── energy_consumption.csv
│           └── Final energy consumption (10000 tons of standard coal).xlsx
├── model
│   └── LLM-Research
│       └── Meta-Llama-3___1-8B-Instruct
│           ├── config.json
│           ├── configuration.json
│           ├── generation_config.json
│           ├── LICENSE
│           ├── model-00001-of-00004.safetensors
│           ├── model-00002-of-00004.safetensors
│           ├── model-00003-of-00004.safetensors
│           ├── model-00004-of-00004.safetensors
│           ├── model.safetensors.index.json
│           ├── original
│           │   ├── consolidated.00.pth
│           │   ├── params.json
│           │   └── tokenizer.model
│           ├── README.md
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── USE_POLICY.md
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
    ├── data.ipynb
    ├── data_processing.py
    ├── direct_ask.py
    ├── extract_feature.py
    ├── extract_hidden_state.py
    ├── ml.py
    ├── __pycache__
    │   ├── data_processing.cpython-310.pyc
    │   ├── direct_ask.cpython-310.pyc
    │   ├── extract_feature.cpython-310.pyc
    │   ├── extract_hidden_state.cpython-310.pyc
    │   ├── ml.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    └── utils.py

18 directories, 49 files
