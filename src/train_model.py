import pandas as pd
import pickle
import os
import json
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

# =========================
# LOAD SCALER & FEATURES
# =========================
with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "rb") as f:
    selected_features_model = pickle.load(f)

print("Scaler et features chargés.")

df = pd.read_csv(DATA_PATH)
y = df["target"]
X = df[selected_features_model]

# Standardisation
X_scaled = scaler.transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val_df = pd.DataFrame(X_val, columns=selected_features_model)

# =========================
# MODEL
# =========================
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=50,     # hyperparamètre ajouté : complexité des arbres
    min_child_samples=20, # hyperparamètre ajouté : évite overfitting
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
)

# =========================
# METRICS
# =========================
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

metrics = {
    "roc_auc": roc_auc_score(y_val, y_pred_proba),
    "precision": precision_score(y_val, y_pred),
    "recall": recall_score(y_val, y_pred),
    "f1_score": f1_score(y_val, y_pred)
}

print("\nÉvaluation sur validation :")
for k, v in metrics.items():
    print(f"{k} : {v:.4f}")

# =========================
# SAVE MODEL & METRICS
# =========================
with open(os.path.join(MODELS_PATH, "best_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODELS_PATH, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Modèle et métriques sauvegardés.")

# =========================
# SHAP EXPLICABILITY
# =========================
print("Calcul SHAP en cours...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_df)

# Summary plot (global importance)
shap.summary_plot(
    shap_values,
    X_val_df,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, "shap_summary.png"))
plt.close()

# Bar plot (importance globale simplifiée)
shap.summary_plot(
    shap_values,
    X_val_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, "shap_bar.png"))
plt.close()

print("SHAP généré :")
print("- models/shap_summary.png")
print("- models/shap_bar.png")
