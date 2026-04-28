
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# 1. Chargement du dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. Suppression de la colonne 'id' (aucune valeur prédictive)
df.drop(['id'], axis=1, inplace=True)

# 3. Nettoyage de la colonne 'gender'
# Il existe une valeur "Other" très rare qui peut faire planter l'encodage binaire
df = df[df['gender'] != 'Other']

# 4. Gestion des valeurs manquantes pour le BMI
# Le dataset contient des "N/A" ou des vides. On utilise la médiane.
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
imputer = SimpleImputer(strategy='median')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# 5. Encodage des variables textuelles (Categorical Encoding)
# On transforme le texte en nombres pour que le MLP puisse les lire
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 6. Feature Scaling (Normalisation) - ÉTAPE CRUCIALE pour le MLP
# Le MLP apprend mal si les échelles sont différentes (ex: Age 0-80 vs Glucose 50-250)
scaler = StandardScaler()
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 7. Sauvegarde du Scaler
# On doit sauvegarder le scaler pour pouvoir normaliser les données du futur patient
joblib.dump(scaler, 'stroke_scaler.pkl')

# 8. Sauvegarde du dataset nettoyé
df.to_csv('healthcare_dataset_cleaned.csv', index=False)

print(" Nettoyage terminé avec succès !")
print(f"Colonnes conservées : {list(df.columns)}")
print(df.head())