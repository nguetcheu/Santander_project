import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import list_columns
import os

# Configuration
DATA_PATH = "./Data/train.csv"
REPORTS_PATH = "./reports/figures"

# Créer le dossier reports s'il n'existe pas
if not os.path.exists(REPORTS_PATH):
    os.makedirs(REPORTS_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable !")

print("="*80)
print("EXPLORATION DES DONNÉES (EDA)")
print("="*80)

# Lister les colonnes
columns = list_columns(DATA_PATH)
print(f"\nNombre de colonnes : {len(columns)}")

# Charger le dataset complet
print("\nChargement du dataset complet...")
df = pd.read_csv(DATA_PATH)
features = [col for col in df.columns if col not in ['ID_code', 'target']]
X = df[features]

print(f"Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# =========================
# APERÇU DU DATASET
# =========================
print("\n" + "="*80)
print("APERÇU DU DATASET")
print("="*80)
print(df.head())

# =========================
# ANALYSE DE LA VARIABLE CIBLE
# =========================
print("\n" + "="*80)
print("DISTRIBUTION DE LA VARIABLE CIBLE")
print("="*80)

target_counts = df['target'].value_counts()
target_pct = df['target'].value_counts(normalize=True) * 100

print(f"\nClasse 0 : {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"Classe 1 : {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"⚠️  Déséquilibre : ratio 1:{target_counts[0]/target_counts[1]:.1f}")

# Visualisation de la distribution de target
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='target', data=df, palette='Set2')
plt.title('Distribution de la Variable Cible', fontsize=14, fontweight='bold')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 500,
            f'{int(height):,}\n({height/len(df)*100:.1f}%)',
            ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, 'target_distribution.png'), dpi=300)
plt.close()
print(f"✓ Graphique sauvegardé : {REPORTS_PATH}/target_distribution.png")

# =========================
# VALEURS MANQUANTES
# =========================
print("\n" + "="*80)
print("VALEURS MANQUANTES")
print("="*80)

missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("✓ Aucune valeur manquante dans le dataset")
else:
    print(missing_values[missing_values > 0])

# =========================
# STATISTIQUES DESCRIPTIVES
# =========================
print("\n" + "="*80)
print("STATISTIQUES DESCRIPTIVES")
print("="*80)

print("\nStatistiques des 10 premières features :")
print(df[features[:10]].describe())

print(f"\n📊 RÉSUMÉ GLOBAL :")
print(f"  • Moyenne générale : {X.mean().mean():.4f}")
print(f"  • Écart-type général : {X.std().mean():.4f}")
print(f"  • Minimum global : {X.min().min():.4f}")
print(f"  • Maximum global : {X.max().max():.4f}")

# =========================
# HISTOGRAMMES - DISTRIBUTION
# =========================
print("\n" + "="*80)
print("HISTOGRAMMES - DISTRIBUTION DES FEATURES")
print("="*80)

print("Génération des histogrammes (20 premières features)...")
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(features[:20]):
    axes[idx].hist(df[col], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(col, fontsize=9)
    axes[idx].tick_params(labelsize=7)

plt.suptitle('Distribution des 20 Premières Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, 'histograms.png'), dpi=300)
plt.close()
print(f"✓ Graphique sauvegardé : {REPORTS_PATH}/histograms.png")

# =========================
# BOXPLOTS - DÉTECTION OUTLIERS
# =========================
print("\n" + "="*80)
print("BOXPLOTS - DÉTECTION DES OUTLIERS")
print("="*80)

# Quantification des outliers avec Z-score
z_scores = np.abs(stats.zscore(X))
outliers_count = (z_scores > 3).sum(axis=0)
outliers_pct = (outliers_count / len(df)) * 100

print(f"\nFeatures avec outliers (Z-score > 3) : {(outliers_count > 0).sum()}/{len(features)}")
print(f"Moyenne d'outliers par feature : {outliers_pct.mean():.2f}%")

# Boxplots
print("Génération des boxplots (20 premières features)...")
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(features[:20]):
    axes[idx].boxplot(df[col], vert=True)
    axes[idx].set_title(col, fontsize=9)
    axes[idx].tick_params(labelsize=7)

plt.suptitle('Boxplots - Détection des Outliers (20 Premières Features)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, 'boxplots.png'), dpi=300)
plt.close()
print(f"✓ Graphique sauvegardé : {REPORTS_PATH}/boxplots.png")

# =========================
# MATRICE DE CORRÉLATION
# =========================
print("\n" + "="*80)
print("MATRICE DE CORRÉLATION")
print("="*80)

print("Calcul de la matrice de corrélation (30 premières features)...")
corr_matrix = df[features[:30]].corr()

# Visualisation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation (30 Premières Features)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, 'correlation_matrix.png'), dpi=300)
plt.close()
print(f"✓ Graphique sauvegardé : {REPORTS_PATH}/correlation_matrix.png")

# Top corrélations
corr_pairs = corr_matrix.unstack()
top_corr = corr_pairs[corr_pairs < 1].abs().sort_values(ascending=False).head(5)

print("\nTop 5 corrélations entre features :")
for (f1, f2), val in top_corr.items():
    print(f"  • {f1} ↔ {f2} : {val:.3f}")

# =========================
# RÉSUMÉ FINAL
# =========================
print("\n" + "="*80)
print("RÉSUMÉ DE L'EXPLORATION")
print("="*80)

print(f"""
📊 DATASET
  • Dimensions : {df.shape[0]:,} lignes × {len(features)} features
  • Type : Toutes variables numériques (anonymisées)
  • Valeurs manquantes : Aucune ✓

🎯 VARIABLE CIBLE
  • Classe 0 : {target_pct[0]:.1f}%
  • Classe 1 : {target_pct[1]:.1f}%
  • Déséquilibre : ratio 1:{target_counts[0]/target_counts[1]:.1f}

📈 STATISTIQUES
  • Moyenne globale : {X.mean().mean():.4f}
  • Écart-type global : {X.std().mean():.4f}
  • Min/Max : [{X.min().min():.2f}, {X.max().max():.2f}]
  • Outliers : ~{outliers_pct.mean():.1f}% par feature

🔗 CORRÉLATIONS
  • Corrélation maximale : {top_corr.iloc[0]:.3f}

📁 GRAPHIQUES GÉNÉRÉS ({REPORTS_PATH}/)
  ✓ target_distribution.png
  ✓ histograms.png
  ✓ boxplots.png
  ✓ correlation_matrix.png
""")

print("✅ Exploration des données terminée avec succès !")