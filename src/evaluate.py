import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "./Data/train.csv"
MODELS_PATH = "./models"

# Vérification des chemins
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} introuvable !")
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

with open(os.path.join(MODELS_PATH, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "rb") as f:
    selected_features_model = pickle.load(f)

print("Modèle, scaler et features chargés.")

df = pd.read_csv(DATA_PATH)
y = df["target"]
X = df[selected_features_model]

# Standardisation
X_scaled = scaler.transform(X)

# Train/Validation split pour évaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# PREDICTIONS
# =========================
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

# =========================
# METRICS
# =========================
roc_auc = roc_auc_score(y_val, y_pred_proba)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(MODELS_PATH, "confusion_matrix.png"))
plt.close()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(MODELS_PATH, "roc_curve.png"))
plt.close()

print("Confusion matrix et ROC curve sauvegardées dans models/")