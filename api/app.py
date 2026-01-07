from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import shap
import os
import json

MODELS_PATH = "./models"
DATA_PATH = "./Data/train.csv"

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL FILES
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

# =========================
# FEATURE MEDIANS
# =========================
df = pd.read_csv(DATA_PATH)
feature_medians = df[model_features].median().to_dict()

# =========================
# MAPPINGS METIER → ML
# =========================

# Champs visibles par l'utilisateur
CREDIT_INPUT_MAPPING = {
    "age": lambda x: x / 100,
    "monthly_income": lambda x: x / 10000,
    "personal_contribution": lambda x: x / 25000,
    "cdi_duration": lambda x: x / 10,
    "current_credits": lambda x: x / 10,
    "loan_duration": lambda x: x / 25,
    "monthly_charges": lambda x: x / 5000,
    "secondary_income": lambda x: x / 5000,
    "bank_seniority": lambda x: x / 10,
    "property_value": lambda x: x / 250000,
    "dependents": lambda x: x / 5,
    "savings_transfers": lambda x: x / 10,
    "avg_monthly_balance": lambda x: x / 10000,
    "subscriptions": lambda x: x / 10,
    "repaid_credits": lambda x: x / 5
}

# Mapping métier → features ML
FEATURE_NAME_MAPPING = {
    "age": "var_174",
    "monthly_income": "var_133",
    "personal_contribution": "var_53",
    "cdi_duration": "var_34",
    "current_credits": "var_99",
    "loan_duration": "var_165",
    "monthly_charges": "var_190",
    "secondary_income": "var_6",
    "bank_seniority": "var_22",
    "property_value": "var_12",
    "dependents": "var_146",
    "savings_transfers": "var_76",
    "avg_monthly_balance": "var_166",
    "subscriptions": "var_78",
    "repaid_credits": "var_21"
}

# Labels lisibles (UI)
FEATURE_LABELS = {
    "var_174": "Âge",
    "var_133": "Revenu mensuel",
    "var_53": "Apport personnel",
    "var_34": "Durée du CDI",
    "var_99": "Crédits en cours",
    "var_165": "Durée du prêt",
    "var_190": "Charges mensuelles",
    "var_6": "Revenus secondaires",
    "var_22": "Ancienneté bancaire",
    "var_12": "Valeur du bien",
    "var_146": "Personnes à charge",
    "var_76": "Virements épargne/an",
    "var_166": "Solde moyen mensuel",
    "var_78": "Abonnements récurrents",
    "var_21": "Crédits remboursés"
}

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "metrics": model_metrics,
        "num_model_features": len(model_features),
        "num_questionnaire_features": len(questionnaire_features),
        "questionnaire_features": questionnaire_features
    })

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Vérifier champs manquants
        missing = [k for k in CREDIT_INPUT_MAPPING if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Créer DataFrame modèle
        X_input = pd.DataFrame(columns=model_features, index=[0])

        # Transformation métier → ML
        for user_field, transform in CREDIT_INPUT_MAPPING.items():
            raw_value = float(data[user_field])
            ml_value = transform(raw_value)
            ml_feature = FEATURE_NAME_MAPPING[user_field]
            X_input.at[0, ml_feature] = ml_value

        # Compléter avec médianes
        for f in model_features:
            if pd.isna(X_input.at[0, f]):
                X_input.at[0, f] = feature_medians[f]

        # Scaling
        X_scaled = scaler.transform(X_input)

        # Prédiction
        proba = model.predict_proba(X_scaled)[0][1]
        score = round(proba * 100, 1)
        decision = "accord" if proba >= 0.7 else "refus"

        response = {
            "score_percent": score,
            "decision": decision
        }

        # Explication en cas de refus
        if decision == "refus":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            shap_client = shap_values[0] if isinstance(shap_values, list) else shap_values

            impacts = pd.DataFrame({
                "feature": model_features,
                "impact": shap_client[0]
            })

            negatives = impacts.sort_values("impact").head(5)

            response["reasons"] = [
                {
                    "feature": FEATURE_LABELS.get(row.feature, row.feature),
                    "impact": round(row.impact, 4)
                }
                for _, row in negatives.iterrows()
            ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
