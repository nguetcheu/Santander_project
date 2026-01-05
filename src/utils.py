# src/utils.py
import pandas as pd

def list_columns(csv_path):
  """
  Affiche les colonnes et le type de chaque colonne d'un CSV.
    
  Args:
    csv_path (str): chemin vers le fichier CSV
    
  Returns:
    list: liste des colonnes
    """
  df = pd.read_csv(csv_path, nrows=18)  
  print("Colonnes du dataset :")
  print(df.dtypes)
  return df.columns.tolist()
