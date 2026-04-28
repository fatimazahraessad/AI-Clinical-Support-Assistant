import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# 1. Chargement (Assure-toi d'avoir les 10 colonnes dans ton CSV)
df = pd.read_csv('healthcare_dataset_cleaned.csv')

# 2. Preprocessing (Nettoyage rapide)
# ... (tes étapes d'encodage pour gender, smoking_status, etc.) ...

# 3. Sélection des 10 features exactement comme dans l'orchestrateur
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
X = df[features]
y = df['stroke']

# 4. Initialisation et entraînement du SCALER sur les 10 colonnes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # C'est ici que le scaler apprend les 10 dimensions

# 5. SMOTE pour équilibrer
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 6. Entraînement du MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
mlp.fit(X_resampled, y_resampled)

# 7. SAUVEGARDE DES DEUX FICHIERS
joblib.dump(mlp, 'stroke_mlp_model_V3.pkl')
joblib.dump(scaler, 'stroke_scaler.pkl') # Ce scaler aura maintenant 10 features !

print("✅ Modèles recréés avec 10 features.")