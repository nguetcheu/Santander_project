import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
from scipy import stats
import pickle
import os

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

df = pd.read_csv(DATA_PATH)
y = df["target"]
X = df.drop(columns=["ID_code", "target"])

# =========================
# Sélection des features via LightGBM
# =========================
# Entraîner un LGBM pour obtenir l'importance des features
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, class_weight="balanced")
lgbm.fit(X, y)

# Récupérer les importances
importances = pd.Series(lgbm.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

# Analyser les outliers dans les features sélectionnées
z_scores = stats.zscore(X)
outliers = (abs(z_scores) > 3).sum(axis=0)
print("Nombre d'outliers par feature :")
print(outliers[outliers > 0])

# 70 features pour le modèle
selected_features_model = importances_sorted.head(70).index.tolist()
with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "wb") as f:
    pickle.dump(selected_features_model, f)
print("70 features sélectionnées pour le modèle :")
print(selected_features_model)

# 15 features pour le questionnaire
selected_features_questionnaire = importances_sorted.head(15).index.tolist()
with open(os.path.join(MODELS_PATH, "selected_features_questionnaire.pkl"), "wb") as f:
    pickle.dump(selected_features_questionnaire, f)
print("70 features sélectionnées pour le questionnaire :")
print(selected_features_questionnaire)

selector = VarianceThreshold(threshold=0.01)
selector.fit(X)
print(f"Features avec variance faible : {sum(~selector.get_support())}")

# =========================
# Scaler sur les 70 features pour le modèle
# =========================
X_selected = X[selected_features_model]
scaler = StandardScaler()
scaler.fit(X_selected)

with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Scaler sauvegardé dans models/scaler.pkl (fit sur 70 features)")