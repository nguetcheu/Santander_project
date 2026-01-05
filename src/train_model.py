import pandas as pd
import pickle
import os
import json
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "rb") as f:
    selected_features = pickle.load(f)

print("Scaler et features chargés.")

df = pd.read_csv(DATA_PATH)
y = df["target"]
X = df[selected_features]

X_scaled = scaler.transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,          # hyperparamètre ajouté : complexité des arbres
    min_child_samples=20,   # hyperparamètre ajouté : évite overfitting
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=50,
    verbose=50
)

y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

roc_auc = roc_auc_score(y_val, y_pred_proba)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

metrics = {
    "roc_auc": roc_auc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

print("\nÉvaluation sur validation :")
for k, v in metrics.items():
    print(f"{k} : {v:.4f}")

with open(os.path.join(MODELS_PATH, "best_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("Modèle sauvegardé dans models/best_model.pkl")

with open(os.path.join(MODELS_PATH, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print("Métriques sauvegardées dans models/metrics.json")
