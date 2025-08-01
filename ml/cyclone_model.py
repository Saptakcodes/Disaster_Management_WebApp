import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os

# ✅ Load dataset
df = pd.read_csv('datasets/storms.csv')

# ✅ Label encoding
df['label'] = df['status'].apply(lambda x: 1 if str(x).lower().strip() in ['hurricane', 'typhoon', 'major hurricane'] else 0)

# ✅ Replace missing values
df.replace([-999, -1998], np.nan, inplace=True)

# ✅ Select features
features = ['lat', 'long', 'wind', 'pressure', 'tropicalstorm_force_diameter', 'hurricane_force_diameter']
df.dropna(subset=features, inplace=True)

# ✅ Feature matrix & target vector
X = df[features]
y = df['label']

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train XGBoost Classifier
model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/cyclone_model.pkl')
print("✅ Model saved to models/cyclone_model.pkl")
