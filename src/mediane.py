import pandas as pd
import json
import pickle

# Charger vos features utilisées par le modèle
with open("./models/selected_features.pkl", "rb") as f:
    model_features = pickle.load(f)

# Charger le gros CSV
df = pd.read_csv("./Data/train.csv")

# Calculer les médianes uniquement pour les features nécessaires
feature_medians = df[model_features].median().to_dict()

# Sauvegarder dans un petit fichier JSON (quelques Ko seulement)
with open("./models/feature_medians.json", "w") as f:
    json.dump(feature_medians, f)

print("Fichier feature_medians.json créé avec succès !")