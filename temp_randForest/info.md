#/training_code folder:
random_forest_model.py: Train the random forest using sklearn, using data/final_combined_data/final_data_with_bow.csv. It saves the model as random_forest_model.joblib， which is not included in the folder at the moment, but it will be generated after you run random_forest_model.py.
You can try to run it but this is what you will see

Loading data...
Dataset shape: (1644, 705)
Number of unique classes: 3
Training set size: 1150 samples
Test set size: 494 samples
Test set saved to data/randomForest/test_set.csv

Training Random Forest model...

Test Accuracy: 0.8927

Classification Report:
              precision    recall  f1-score   support

       Pizza       0.88      0.89      0.89       140
    Shawarma       0.91      0.90      0.91       188
       Sushi       0.88      0.89      0.88       166

    accuracy                           0.89       494
   macro avg       0.89      0.89      0.89       494
weighted avg       0.89      0.89      0.89       494

random_forest_extractor.py: loads and saves the model from random_forest_model.joblib by extracting all parameters, then converts it to JSON format and stores it in rf_params.json. The .json file is currently under the /code folder, which is necessary for prep.py to run. 


/code folder:

Contains all the code needed for pred.py to run correctly on markus.
Locally, we need a raw file for pred.py to run.

Example usage: python pred.py data/cleaned_data_combined_modified.csv
Prep.py will also save its prediction into a new csv file predictions_aligned.csv, which is useful for testing
i.e., compare the label of the initial raw csv file and the predictions_aligned.csv line by line to calculate the accuracy. This can be done by a simple python script. I didn't include this script in the folder to keep it clean.

Below is the accuracy of running pred.py on the raw file data/cleaned_data_combined_modified.csv. 

Results:
Total lines compared: 1644
Correct predictions: 1555
Accuracy: 0.9459 (94.59%)

Apparently this is not realistic accuracy since the model was trained on the same piece of data. But at least it shows that the model is running as expected.



If you have any other questions, please let me know

Tianlei


