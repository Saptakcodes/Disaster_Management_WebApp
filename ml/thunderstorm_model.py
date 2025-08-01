import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Example: Load your dataset (replace with your actual data source)
# Assume 'data.csv' has columns for features + 'weather_code' (target)
data = pd.read_csv('datasets/thunderstorm_dataset.csv')

# Define feature columns (your input array)
features = [
    "temperature", "dew_point", "humidity", "pressure",
    "wind_speed", "wind_direction", "cloud_cover", "precipitation",
    "uvi", "visibility", "elevation"
]

# Target column
target = 'thunderstorm'  # 1 for thunderstorm, 0 for no thunderstorm

# Create a binary target based on weather_code (assuming 200 indicates a thunderstorm)
data[target] = (data['weather_code'] == 200).astype(int)

# Prepare X and y
X = data[features]
y = data[target]

# Optional: preprocess wind_direction (categorical cyclical encoding)
X['wind_dir_sin'] = np.sin(np.deg2rad(X['wind_direction']))
X['wind_dir_cos'] = np.cos(np.deg2rad(X['wind_direction']))
X = X.drop('wind_direction', axis=1)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#file dumped
import joblib
joblib.dump(model, 'thunderstorm_classifier.pkl')
