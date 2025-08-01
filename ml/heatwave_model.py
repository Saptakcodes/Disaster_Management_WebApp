# 1. Importing Required Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 2. Load Dataset
df = pd.read_csv('datasets/heatwave_dataset.csv')
print(df.head())
print(df.tail())

# 3. Data Preprocessing

# Fill missing values for continuous columns
continuous_cols = [
    'wind_speed', 'cloud_cover', 'precipitation_probability', 
    'pressure_surface_level', 'dew_point', 'uv_index', 'visibility',
    'max_temperature', 'min_temperature', 'max_humidity', 'min_humidity',
    'rainfall', 'snowfall', 'latitude', 'longitude'
]
for col in continuous_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Drop unwanted or incomplete columns
df = df.dropna(subset=['heatwave'])
df = df.drop(columns=[col for col in ['solar_radiation', 'date'] if col in df.columns])

# Convert datetime column if still present
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    df = df.drop(columns=['date'])

# Ensure binary fields exist and are clean (0 or 1)
for col in ['historical_event_present', 'urban_heat_effect']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = 0  # Fallback if not found

# Final Null Check
print("Missing values after processing:\n", df.isnull().sum())

# 4. Feature Selection
selected_features = [
    'max_temperature', 
    'min_temperature',
    'max_humidity', 
    'min_humidity',
    'precipitation_probability',
    'uv_index',
    'visibility',
    'historical_event_present',
    'urban_heat_effect'
]
X = df[selected_features]
y = df['heatwave']

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 7. Model Training
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# 8. Model Evaluation
y_pred = rfc.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importances
importances = rfc.feature_importances_
feature_names = selected_features
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(importances)), importances[sorted_idx], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
plt.tight_layout()
plt.show()

# 9. Save Model
joblib.dump(rfc, 'rfc_heatwave_model.pkl')

