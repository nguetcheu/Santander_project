# Santander Project - Machine Learning End-to-End

## 📝 Description du projet

Ce projet vise à construire un pipeline complet de Machine Learning pour prédire la variable `target` sur le dataset Santander.  
L’objectif est de créer un modèle performant tout en permettant l’explicabilité, l’évaluation et la mise à disposition via une API web.

Le pipeline inclut :  
- Téléchargement et manipulation des datasets Kaggle  
- Analyse exploratoire (EDA)  
- Préprocessing et sélection de features  
- Gestion de classes déséquilibrées  
- Entraînement et comparaison de modèles (LightGBM, Random Forest, Régression Linéaire)  
- Optimisation d’hyperparamètres  
- Création d’une API REST avec Flask  
- Interface web avec HTML/CSS  

---

## 📁 Structure du projet

santander_project/
├── data/ # Jeux de données
│ ├── train.csv # Training set
│ └── test.csv # Test set
├── src/ # Scripts Python pour le pipeline ML
│ ├── eda.py # Analyse exploratoire (logs + graphiques PNG)
│ ├── preprocessing.py # Sélection de features + scaler
│ ├── train_model.py # Entraînement et sauvegarde du modèle final
│ ├── evaluate.py # Évaluation des modèles (ROC, confusion matrix)
│ └── utils.py # Fonctions utilitaires (chargement, métriques)
├── models/ # Modèles et objets ML sauvegardés
│ ├── best_model.pkl # Meilleur modèle entraîné
│ ├── scaler.pkl # StandardScaler pour les features
│ ├── selected_features.pkl # Liste des features retenues pour le modèle
│ └── selected_features_questionnaire.pkl # Features pour questionnaire
├── api/ # API Flask
│ ├── app.py # Application Flask
│ └── requirements.txt # Librairies nécessaires pour l’API
├── frontend/ # Interface web
│ ├── index.html # Page principale
│ └── styles.css # Styles CSS
├── reports/ # Graphiques et rapports automatiques
│ └── figures/ # Graphiques EDA et évaluation
├── requirements.txt # Librairies Python globales (scikit-learn, pandas, etc.)
├── README.md # Documentation du projet
└── .gitignore # Fichiers à ignorer (models/, pycache, etc.)

---

## 🧪 Dataset

- **train.csv** : dataset d’entraînement avec la colonne `target`.  
- **test.csv** : dataset de test pour prédictions.  
- Le projet prend en charge les datasets déséquilibrés (classe `target` minoritaire).

---

## ⚙️ Pipeline Machine Learning

1. **EDA** (`src/eda.py`)  
   - Analyse des distributions, corrélations et visualisations.
   
2. **Préprocessing** (`src/preprocessing.py`)  
   - Standardisation des features (`StandardScaler`)  
   - Sélection des features importantes pour le modèle et le questionnaire

3. **Entraînement du modèle** (`src/train_model.py`)  
   - Modèle principal : **LightGBM**  
   - Hyperparamètres optimisés :  
     ```text
     n_estimators=2000, learning_rate=0.05, max_depth=8, num_leaves=50, 
     min_child_samples=20, class_weight="balanced", random_state=42
     ```
   - Meilleur score ROC-AUC obtenu : **0.8610** avec 70 features

4. **Explicabilité** (`src/explain_model.py`)  
   - Utilisation de **SHAP** pour visualiser l’importance des features  
   - Graphiques générés : `shap_summary.png` et `shap_bar.png`  

5. **Évaluation** (`src/evaluate.py`)  
   - Métriques : ROC-AUC, Precision, Recall, F1-Score  
   - Graphiques : matrice de confusion, courbe ROC  

---

## 🌐 API Flask

- **api/app.py** : API REST permettant de :
  - Recevoir des données JSON
  - Retourner les prédictions du modèle
  - Afficher éventuellement des informations sur les features ou explications SHAP

- Dépendances : `Flask`, `pandas`, `scikit-learn`, `lightgbm`, `pickle`

---

## 🖥️ Frontend

- Simple interface web en **HTML et CSS** pour interagir avec l’API.  
- Exemple : formulaire pour saisir les valeurs des features et obtenir la prédiction.

---

## 📈 Évaluation

- Meilleur modèle : **LightGBM**  
- **ROC-AUC** : 0.8610  
- **Precision** : 0.3607  
- **Recall** : 0.6672  
- **F1-Score** : 0.4683  
- Graphiques sauvegardés dans `models/` : matrice de confusion et courbe ROC

---

## 🔧 Installation

1. Cloner le projet :  

git clone <repo_url>
cd santander_project
Créer un environnement Python et installer les dépendances :

2. Cloner le projet :  
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt


3. Lancer les scripts :

python src/preprocessing.py
python src/train_model.py
python src/evaluate.py
python src/explain_model.py


4. Lancer l’API Flask :

`cd api`
python app.py

## 📝 Notes

Les scripts sont indépendants pour permettre des tests modulaires.

Le modèle LightGBM est utilisé pour la prédiction finale et SHAP pour l’explicabilité.

Les features du questionnaire sont séparées pour usage simplifié.

## 📂 Résultats

Modèle entraîné : models/best_model.pkl

Features sélectionnées : models/selected_features.pkl

SHAP summary et bar plot : models/shap_summary.png, models/shap_bar.png

Évaluation : models/metrics.json, models/confusion_matrix.png, models/roc_curve.png