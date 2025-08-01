import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Load and clean dataset
df = pd.read_csv('datasets/Algerian_forest_fires_dataset_CLEANED.csv')
df.drop(['day', 'month', 'year'], axis=1, inplace=True)
df['Classes'] = np.where(df['Classes'] == 'not fire', 0, 1)

# Define features and target
X = df.drop('FWI', axis=1)
y = df['FWI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Correlation analysis
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_corr.add(corr_matrix.columns[i])
    return col_corr

corr_features = correlation(X_train, 0.75)
X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)

# Standardization
def scaler_standard(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scaler_standard(X_train, X_test)

# Baseline Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print("Random Forest Regressor")
print("R2 Score:", round(r2_score(y_test, rf_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, rf_pred), 4))

# Randomized Hyperparameter Tuning
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

rf_reg = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_reg, param_distributions=param_grid,
                                   n_iter=20, cv=5, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train_scaled, y_train)
best_rf = random_search.best_estimator_

# Evaluation
best_pred = best_rf.predict(X_test_scaled)
print("Random Forest Tuned")
print("R2 Score:", round(r2_score(y_test, best_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, best_pred), 4))

# Feature Importances
importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', edgecolor='black')
plt.title("Feature Importances (Random Forest)", fontsize=15, weight='bold')
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# Optional: remove less important features and retrain
drop_cols = ['Rain', 'Region', 'RH']
X_train_opt = X_train.drop(columns=[col for col in drop_cols if col in X_train.columns])
X_test_opt = X_test.drop(columns=[col for col in drop_cols if col in X_test.columns])
X_train_opt_scaled, X_test_opt_scaled = scaler_standard(X_train_opt, X_test_opt)

best_rf.fit(X_train_opt_scaled, y_train)
final_pred = best_rf.predict(X_test_opt_scaled)

print("Optimized Feature Set Evaluation:")
print("R2 Score:", round(r2_score(y_test, final_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, final_pred), 4))
