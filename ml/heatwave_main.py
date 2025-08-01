#1.loading all libraries and packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib

#2.loading dataset
df=pd.read_csv('datasets/heatwave_dataset.csv')
print(df.head())
print(df.tail())

#3.data preprocessing
# Check for missing values
print(df.isnull().sum())

# Drop or fill missing values (based on amount)
df['rainfall'] = df['rainfall'].fillna(df['rainfall'].mean())

df = df.dropna(subset=['heatwave'])
df = df.drop(columns=['solar_radiation'])
# Fill location-based missing (if minor, else consider dropping rows)
df['latitude'] = df['latitude'].fillna(df['latitude'].mean())
df['longitude'] = df['longitude'].fillna(df['longitude'].mean())

# Fill continuous environmental data with mean
for col in ['wind_speed', 'cloud_cover', 'precipitation_probability', 
            'pressure_surface_level', 'dew_point', 'uv_index', 'visibility',
            'max_temperature', 'min_temperature', 'max_humidity', 'min_humidity']:
    df[col] = df[col].fillna(df[col].mean())

# Rainfall/Snowfall - likely to be 0 if missing
df['rainfall'] = df['rainfall'].fillna(0)
df['snowfall'] = df['snowfall'].fillna(0)
df = df.dropna(subset=['date'])  # No imputation for datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

# Extract date features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour

df = df.drop(columns=['date'])  # drop original date

print(df.isnull().sum())  # Should all be 0

X = df.drop('heatwave', axis=1)
y = df['heatwave']

#4.z-score standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#5.test-train split 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

#6.model train
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

#7.model prediction
y_pred = rfc.predict(X_test)

#8.model evaluation
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

importances = rfc.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx], rotation=90)
plt.tight_layout()
plt.show()


#9.model stored and dumped
joblib.dump(rfc, 'random_forest_heatwave_model.pkl')






