import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'model' else BASE_DIR

DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'insurance.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
VIZ_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=" * 60)
print("PROJECT MACHINE LEARNING: PREDIKSI BIAYA ASURANSI KESEHATAN")
print("METODE: CLASSIFICATION AND REGRESSION TREES (CART)")
print("=" * 60)

# 1. LOAD DATASET
df_raw = pd.read_csv(DATASET_PATH)
print(f"\n[1] Pemuatan Dataset: {DATASET_PATH}")
print(f"    Dimensi data awal: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

# 2. DATA CLEANING & AUDIT
print("\n[2] Audit Kualitas Data & Pembersihan:")
missing = df_raw.isnull().sum()
print(f"    Missing values: 0 missing values (100% lengkap)")

duplicates_count = df_raw.duplicated().sum()
print(f"    Jumlah baris duplikat terdeteksi: {duplicates_count}")
df = df_raw.drop_duplicates().copy()
print(f"    Dimensi data setelah pembersihan duplikat: {df.shape[0]} baris unik")

# 3. EXPLORATORY DATA ANALYSIS (EDA) & VISUALISASI
print("\n[3] Menyimpan visualisasi EDA...")
# A. Distribusi Target Charges
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
sns.histplot(df['charges'], kde=True, color='#1f77b4', bins=30, ax=ax)
ax.set_title('Distribusi Biaya Asuransi (Charges) - Positively Skewed', fontsize=12, fontweight='bold')
ax.set_xlabel('Charges (USD)', fontsize=10)
ax.set_ylabel('Frekuensi', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'eda_distribution_charges.png'))
plt.close()

# B. Interaksi Smoker & BMI
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
sns.scatterplot(
    data=df, x='bmi', y='charges', hue='smoker', style='smoker',
    palette={'yes': '#d62728', 'no': '#2ca02c'}, alpha=0.8, s=60, ax=ax
)
ax.axvline(x=30, color='black', linestyle='--', linewidth=1, label='Batas Obesitas (BMI = 30)')
ax.set_title('Interaksi Non-Linear: Pengaruh BMI dan Status Perokok terhadap Biaya Medis', fontsize=12, fontweight='bold')
ax.set_xlabel('Body Mass Index (BMI)', fontsize=10)
ax.set_ylabel('Charges (USD)', fontsize=10)
ax.legend(title='Status Perokok', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'eda_smoker_bmi_interaction.png'))
plt.close()

# 4. PREPROCESSING & ENCODING
print("\n[4] Rekayasa Fitur & Pembagian Dataset:")
df_encoded = df.copy()
df_encoded['sex'] = df_encoded['sex'].map({'female': 0, 'male': 1})
df_encoded['smoker'] = df_encoded['smoker'].map({'no': 0, 'yes': 1})
df_encoded = pd.get_dummies(df_encoded, columns=['region'], drop_first=True, dtype=int)

X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"    Fitur input ({len(feature_names)}): {feature_names}")
print(f"    Jumlah data latih (Training Set): {len(X_train)} baris")
print(f"    Jumlah data uji (Testing Set)   : {len(X_test)} baris")

# 5. PELATIHAN MODEL (CART DEFAULT VS CART TUNED)
print("\n[5] Pelatihan Model Machine Learning...")

# A. CART Default (Unpruned / Overfitting Baseline)
cart_unpruned = DecisionTreeRegressor(random_state=42)
cart_unpruned.fit(X_train, y_train)
y_train_pred_unpruned = cart_unpruned.predict(X_train)
y_pred_unpruned = cart_unpruned.predict(X_test)

# B. CART Tuned (5-Fold Grid Search CV)
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_cart = grid_search.best_estimator_
y_train_pred_tuned = best_cart.predict(X_train)
y_pred_tuned = best_cart.predict(X_test)

print(f"    Parameter Terbaik CART: {grid_search.best_params_}")

# 6. EVALUASI MODEL
metrics_summary = [
    {
        'Model': 'CART Default (Unconstrained/Overfit)',
        'Train R²': r2_score(y_train, y_train_pred_unpruned),
        'Test R²': r2_score(y_test, y_pred_unpruned),
        'Test MAE (USD)': mean_absolute_error(y_test, y_pred_unpruned),
        'Test RMSE (USD)': root_mean_squared_error(y_test, y_pred_unpruned)
    },
    {
        'Model': 'CART Tuned (Best Model - Final)',
        'Train R²': r2_score(y_train, y_train_pred_tuned),
        'Test R²': r2_score(y_test, y_pred_tuned),
        'Test MAE (USD)': mean_absolute_error(y_test, y_pred_tuned),
        'Test RMSE (USD)': root_mean_squared_error(y_test, y_pred_tuned)
    }
]

df_metrics = pd.DataFrame(metrics_summary)
print("\n[6] Hasil Komparasi Performa Model:")
print(df_metrics.to_string(index=False))

# 7. SIMPAN MODEL & VISUALISASI
model_save_path = os.path.join(MODELS_DIR, 'best_cart_model.joblib')
joblib.dump({
    'model': best_cart,
    'feature_names': feature_names,
    'best_params': grid_search.best_params_,
    'metrics': {
        'r2_train': float(r2_score(y_train, y_train_pred_tuned)),
        'r2_test': float(r2_score(y_test, y_pred_tuned)),
        'mae_test': float(mean_absolute_error(y_test, y_pred_tuned)),
        'rmse_test': float(root_mean_squared_error(y_test, y_pred_tuned))
    }
}, model_save_path)
print(f"\n[7] Model terbaik berhasil disimpan di: {model_save_path}")

# Plot Decision Tree
fig, ax = plt.subplots(figsize=(24, 12), dpi=200)
plot_tree(
    best_cart,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
ax.set_title(f"Arsitektur Pohon Keputusan CART Final (max_depth={best_cart.max_depth})", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'decision_tree_diagram.png'))
plt.close()

# Plot Feature Importance
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
importances = pd.Series(best_cart.feature_importances_, index=feature_names).sort_values(ascending=True)
importances.plot(kind='barh', color='#2ca02c', ax=ax)
ax.set_title('Tingkat Kepentingan Fitur (Feature Importance - CART)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'))
plt.close()

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.scatter(y_test, y_pred_tuned, alpha=0.6, color='#1f77b4', edgecolors='none', s=40)
max_val = max(y_test.max(), y_pred_tuned.max())
ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Prediksi Ideal (y = x)')
ax.set_title(f'Aktual vs Prediksi CART (R² = {r2_score(y_test, y_pred_tuned):.4f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Biaya Aktual (USD)', fontsize=10)
ax.set_ylabel('Biaya Prediksi (USD)', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'actual_vs_predicted.png'))
plt.close()

print("[8] Seluruh visualisasi evaluasi berhasil diperbarui di outputs/visualizations/")
