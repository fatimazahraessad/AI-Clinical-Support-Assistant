
import joblib
import pandas as pd
import numpy as np

# Liste tes fichiers ici
model_files = ['stroke_mlp_model.pkl', 'stroke_mlp_model_V3.pkl']
scaler_file = 'stroke_scaler.pkl'

# Un patient type (22 ans, hypertension, etc.)
test_data = pd.DataFrame([{
    'gender': 1, 'age': 22.0, 'hypertension': 1, 'heart_disease': 1,
    'ever_married': 1, 'work_type': 2, 'Residence_type': 1,
    'avg_glucose_level': 180.0, 'bmi': 30.0, 'smoking_status': 3
}])

def test_models():
    try:
        scaler = joblib.load(scaler_file)
        input_scaled = scaler.transform(test_data)
        print("--- [OK] Scaler chargé et données transformées ---")
        
        for m_file in model_files:
            try:
                model = joblib.load(m_file)
                prob = model.predict_proba(input_scaled)[0][1]
                print(f"--- [SUCCESS] {m_file} fonctionne ! Score : {prob*100:.2f}% ---")
            except Exception as e:
                print(f"--- [ECHEC] {m_file} : {e} ---")
    except Exception as e:
        print(f"--- [ERREUR SCALER] : {e} ---")

if __name__ == "__main__":
    test_models()