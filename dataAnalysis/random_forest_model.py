import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
test_data.to_csv('data/randomForest/test_set.csv', index=False)
print("Test set saved to data/randomForest/test_set.csv")

# Train the Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,     # Number of trees in the forest
    max_depth=None,       # Maximum depth of trees (None means unlimited)
    min_samples_split=2,  # Minimum samples required to split a node
    min_samples_leaf=1,   # Minimum samples required at a leaf node
    max_features='sqrt',  # Number of features to consider for best split
    bootstrap=True,       # Use bootstrap samples when building trees
    n_jobs=-1,            # Use all CPU cores
    random_state=42       # For reproducibility
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
feature_importance = model.feature_importances_
feature_names = X.columns
top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:10]
print("\nTop 10 important features:")
for importance, feature in top_features:
    print(f"{feature}: {importance:.4f}")

# Save the model and label encoder for later use
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("\nModel and label encoder saved.") 