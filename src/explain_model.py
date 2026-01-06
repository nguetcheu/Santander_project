import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

# =========================
# LOAD MODEL, SCALER & FEATURES
# =========================
with open(os.path.join(MODELS_PATH, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "rb") as f:
    selected_features_model = pickle.load(f)

print("Modèle, scaler et features chargés.")

df = pd.read_csv(DATA_PATH)
X = df[selected_features_model]

# Standardisation
X_scaled = scaler.transform(X)
X_df = pd.DataFrame(X_scaled, columns=selected_features_model)

# =========================
# SHAP EXPLICABILITY
# =========================
print("Calcul SHAP en cours...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)

# Summary plot (importance globale)
shap.summary_plot(shap_values, X_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, "shap_summary.png"))
plt.close()

# Bar plot (importance globale simplifiée)
shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, "shap_bar.png"))
plt.close()

print("SHAP généré et sauvegardé :")
print("- models/shap_summary.png")
print("- models/shap_bar.png")
