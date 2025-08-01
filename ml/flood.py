import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
df = pd.read_csv("datasets/flood.csv")

# Create a binary column for flood prediction
df['Flood'] = df['FloodProbability'].apply(lambda x: 1 if x >= 0.5 else 0)

# Features and labels
X = df.drop(['FloodProbability', 'Flood'], axis=1)
y = df['Flood']

# Print the order of features used
print("\nFeature Order Used in Model:")
print(list(X.columns))

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Example: Predict using one input row from test set
sample_input = X_test.iloc[[0]]
sample_prediction = model.predict(sample_input)
print("\nPredicted Flood Risk:", "Yes" if sample_prediction[0] == 1 else "No")

# Save the trained model
joblib.dump(model, 'flood_prediction_model.pkl')

# Feature importance plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Feature Importances for Flood Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
