# /training_code folder:
## random_forest_model.py:
Train the random forest using sklearn, using data/final_combined_data/final_data_with_bow.csv. It saves the model as random_forest_model.joblib， which is not included in the folder at the moment, but it will be generated after you run random_forest_model.py.
You can try to run it, but below is what you will see:

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

## random_forest_extractor.py: 
loads and saves the model from random_forest_model.joblib by extracting all parameters, then converts it to JSON format and stores it in rf_params.json. The .json file is currently under the /code folder, which is necessary for prep.py to run. 



# /code folder:

Contains all the code needed for pred.py to run correctly on markus.


## rf_params.json:
contains the parameters of the random forest trained by random_forest_model.py, extracted by random_forest_extractor.py. prep.py directly relies on this file.

## feature_name.csv:
Contains all predictor variables built from the training set. Because of the bag-of-words feature, using different input files with different movie titles will result in different counts of predictor variables. This is a big problem because, if we process the train file and the test file separately, there will be a mismatch in the predictor variables' index between the two processed files, which makes the decision trees choose the wrong variable when doing prediction on the processed test file.

Thus, all the predictor variables generated from the train file are stored in a separate CSV file, and prep.py processes the test file according to these predictors. 
For any test file, unknown words will be ignored (not included as features), and missing features are filled with zeros. Thus, no matter what the test file is, the same number of predictor variables(as well as the index) is guaranteed. 

pred.py directly relies on this file when processing the test file.

## pred.py
Main file of the folder.

Locally, we need a raw file for pred.py to run.

Example usage: python pred.py data/cleaned_data_combined_modified.csv
Prep.py will save its prediction into a new CSV file predictions_aligned.csv, which is useful for testing
i.e., compare the label of the initial raw CSV file and the predictions_aligned.csv line by line to calculate the accuracy. This can be done by a simple python script. I didn't include this script in the folder to keep it clean.

Below is the accuracy of running pred.py on the raw file data/cleaned_data_combined_modified.csv. 

Results:
Total lines compared: 1644
Correct predictions: 1555
Accuracy: 0.9459 (94.59%)

Apparently this is not realistic accuracy since the model was trained on the same piece of data. But at least it shows that the model is running as expected.



If you have any other questions, please let me know

Tianlei


