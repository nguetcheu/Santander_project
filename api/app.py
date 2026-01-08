from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import shap
import os
import json

MODELS_PATH = "./models"
HISTORY_FILE = os.path.join(MODELS_PATH, "history.json")
MEDIANS_FILE = os.path.join(MODELS_PATH, "feature_medians.json")

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL, SCALER, FEATURES, METRICS & MEDIANS
# =========================
with open(os.path.join(MODELS_PATH, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features.pkl"), "rb") as f:
    model_features = pickle.load(f)

with open(os.path.join(MODELS_PATH, "selected_features_questionnaire.pkl"), "rb") as f:
    questionnaire_features = pickle.load(f)

with open(os.path.join(MODELS_PATH, "metrics.json"), "r") as f:
    model_metrics = json.load(f)

# CHARGEMENT DES MÉDIANES (Remplacent le CSV de 300Mo)
if os.path.exists(MEDIANS_FILE):
    with open(MEDIANS_FILE, "r") as f:
        feature_medians = json.load(f)
else:
    # Optionnel: lever une erreur si le fichier est absent au démarrage
    raise FileNotFoundError(f"Le fichier {MEDIANS_FILE} est requis pour le déploiement.")

HUMAN_TO_MODEL = {
    "age": ("var_174", 100),
    "monthly_income": ("var_133", 10000),
    "personal_contribution": ("var_53", 25000),
    "cdi_duration": ("var_34", 10),
    "current_credits": ("var_99", 10),
    "loan_duration": ("var_165", 25),
    "monthly_charges": ("var_190", 5000),
    "secondary_income": ("var_6", 5000),
    "bank_seniority": ("var_22", 10),
    "property_value": ("var_12", 250000),
    "dependents": ("var_146", 5),
    "savings_transfers": ("var_76", 10),
    "avg_monthly_balance": ("var_166", 10000),
    "subscriptions": ("var_78", 10),
    "repaid_credits": ("var_21", 5)
}

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/historique")
def historique():
    return render_template("historique.html")

@app.route("/api/historique")
def historique_api():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        return jsonify({
            "total_requests": len(history),
            "history": history[::-1]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        missing_inputs = [k for k in HUMAN_TO_MODEL.keys() if k not in data]
        if missing_inputs:
            return jsonify({"error": f"Missing inputs: {missing_inputs}"}), 400

        # Créer le DataFrame
        X_input = pd.DataFrame(columns=model_features, index=[0])

        # Remplir les features du questionnaire
        for human_name, (var_name, divisor) in HUMAN_TO_MODEL.items():
            X_input.at[0, var_name] = data[human_name] / divisor

        # Remplir le reste avec les médianes chargées depuis le JSON
        for f in model_features:
            if pd.isna(X_input.at[0, f]):
                X_input.at[0, f] = feature_medians[f]

        # Scaler + Prédiction
        X_scaled = scaler.transform(X_input)
        proba = model.predict_proba(X_scaled)[0][1]
        score_percent = round(proba * 100, 1)
        decision = "accord" if proba >= 0.7 else "refus"

        response = {"score_percent": score_percent, "decision": decision}

        # SHAP
        if decision == "refus":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            shap_values_client = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            feature_impacts = pd.DataFrame({
                "feature": model_features,
                "value": X_input.values[0],
                "impact": shap_values_client[0] if len(shap_values_client.shape) > 1 else shap_values_client
            })
            top_negatives = feature_impacts.sort_values("impact").head(5)
            response["top_negative_features"] = top_negatives.to_dict(orient="records")
            
        # Historique
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "score_percent": score_percent,
            "inputs": data
        }

        if decision == "refus":
            history_entry["top_negative_features"] = response.get("top_negative_features", [])

        history.append(history_entry)

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
