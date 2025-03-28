training_code folder:
random_forest_model.py: train the random forest using sklearn, using data/final_combined_data/final_data_with_bow.csv. It saves the model as random_forest_model.joblib

random_forest_extractor.py: loads and saved model from random_forest_model.joblib by extracting all parameters, then convert it to JSON format stored in rf_params.json


code folder:

Contains all code needed for pred.py to run correctly on markus.
locally, need a raw file for pred.py to run.

For example: python pred.py data/cleaned_data_combined_modified.csv


