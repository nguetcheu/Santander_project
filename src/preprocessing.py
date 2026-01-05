import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

df = pd.read_csv(DATA_PATH)

# Séparer X et y
y = df["target"]
X = df.drop(columns=["ID_code", "target"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sauvegarder le scaler
with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Scaler sauvegardé dans models/scaler.pkl")

# =========================
# Sélection des 33 features via régression logistique
# =========================
lr = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
lr.fit(X_scaled, y)

# Récupérer les coefficients
coefs = np.abs(lr.coef_[0])
features = X.columns
feature_coef = list(zip(features, coefs))

# Trier par importance décroissante
feature_coef_sorted = sorted(feature_coef, key=lambda x: x[1], reverse=True)

# Garder les 33 plus importantes
selected_features = [feat for feat, coef in feature_coef_sorted[:33]]
print("33 features sélectionnées :")
print(selected_features)

# Sauvegarder les features sélectionnées
with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "wb") as f:
    pickle.dump(selected_features, f)
print("Features sauvegardées dans models/selected_features.pkl")
