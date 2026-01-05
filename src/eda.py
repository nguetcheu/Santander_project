import pandas as pd
from utils import list_columns
import os

DATA_PATH = "./Data/train.csv"  

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")

columns = list_columns(DATA_PATH)

df = pd.read_csv(DATA_PATH, nrows=1000)

print("\nAperçu du dataset (5 premières lignes) :")
print(df.head())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())
