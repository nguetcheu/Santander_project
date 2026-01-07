from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template
import pandas as pd
import numpy as np
import pickle
import shap
import os
import json

MODELS_PATH = "./models"
DATA_PATH = "./Data/train.csv"  # Pour calculer les médianes si besoin

app = Flask(__name__)
CORS(app) 

# =========================
# LOAD MODEL, SCALER, FEATURES, METRICS
# =========================
model_file = os.path.join(MODELS_PATH, "best_model.pkl")
scaler_file = os.path.join(MODELS_PATH, "scaler.pkl")
features_file = os.path.join(MODELS_PATH, "selected_features.pkl")
questionnaire_file = os.path.join(MODELS_PATH, "selected_features_questionnaire.pkl")
metrics_file = os.path.join(MODELS_PATH, "metrics.json")

with open(model_file, "rb") as f:
    model = pickle.load(f)

with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

with open(features_file, "rb") as f:
    model_features = pickle.load(f)

with open(questionnaire_file, "rb") as f:
    questionnaire_features = pickle.load(f)

with open(metrics_file, "r") as f:
    model_metrics = json.load(f)

# Charger les médianes des features
df = pd.read_csv(DATA_PATH)
feature_medians = df[model_features].median().to_dict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/model-info", methods=["GET"])
def model_info():
    info = {
        "num_model_features": len(model_features),
        "num_questionnaire_features": len(questionnaire_features),
        "questionnaire_features": questionnaire_features,
        "metrics": model_metrics
    }
    return jsonify(info), 200

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        # Vérifier que toutes les features du questionnaire sont présentes
        missing_features = [f for f in questionnaire_features if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Créer un DataFrame pour toutes les features du modèle
        X_input = pd.DataFrame(columns=model_features, index=[0])

        # Remplir les 15 features du questionnaire
        for f in questionnaire_features:
            X_input.at[0, f] = data[f]

        # Remplir le reste avec la médiane
        for f in model_features:
            if pd.isna(X_input.at[0, f]):
                X_input.at[0, f] = feature_medians[f]

        # Appliquer scaler
        X_scaled = scaler.transform(X_input)

        # Prédiction probabilité
        proba = model.predict_proba(X_scaled)[0][1]
        score_percent = round(proba * 100, 1)

        # Déterminer accord/refus
        decision = "accord" if proba >= 0.7 else "refus"

        response = {"score": score_percent, "prediction": decision}

        # Si refus, expliquer les variables défavorables avec SHAP
        if decision == "refus":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            shap_values_client = shap_values[0] if isinstance(shap_values, list) else shap_values

            # Prendre les 5 features les plus négatives
            feature_impacts = pd.DataFrame({
                "feature": model_features,
                "value": X_input.values[0],
                "impact": shap_values_client[0] if isinstance(shap_values_client, np.ndarray) else shap_values_client
            })
            top_negatives = feature_impacts.sort_values("impact").head(5)
            response["top_negative_features"] = top_negatives.to_dict(orient="records")

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
