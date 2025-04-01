import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
 
# Load the dataset
print("Loading data...")
df = pd.read_csv("data/final_combined_data/final_data_with_bow.csv")
print(f"Dataset shape: {df.shape}")

# Encode the labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
print(f"Number of unique classes: {len(label_encoder.classes_)}")

# Drop the ID column
df = df.drop(columns=['id'])

# Split into features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Save test set to CSV with original text labels
test_data = X_test.copy()
test_data['Label'] = label_encoder.inverse_transform(y_test)  # Convert back to original labels
test_data.to_csv('data/randomForest/test_set_hyper.csv', index=False)
print("Test set saved to data/randomForest/test_set_hyper.csv")

# Train the Random Forest model
print("\nTraining Random Forest model...")
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

model = RandomForestClassifier(
    bootstrap=True,       # Use bootstrap samples when building trees
    n_jobs=-1,            # Use all CPU cores
    random_state=42       # For reproducibility
)
grid_search = GridSearchCV(estimator=model, 
                         param_grid=param_grid,
                         cv=5,
                         n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
feature_importance = best_model.feature_importances_
feature_names = X.columns
top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:10]
print("\nTop 10 important features:")
for importance, feature in top_features:
    print(f"{feature}: {importance:.4f}")

# Save the model and label encoder for later use
joblib.dump(best_model, 'random_forest_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("\nModel and label encoder saved.") 