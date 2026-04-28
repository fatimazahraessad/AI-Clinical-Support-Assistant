import os
import base64
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_symptoms_from_image(image_path):
    try:
        if not os.path.exists(image_path):
            return {"error": "Fichier introuvable"}

        # Encodage de l'image
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        print(f"--- Analyse de l'IRM : {os.path.basename(image_path)} ---")
        
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this MRI image. List only visible clinical signs or abnormalities in a JSON array. No diagnosis. No conversational text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
                        }
                    ]
                }
            ],
            temperature=0,
        )

        # Récupération du texte brut
        raw_content = completion.choices[0].message.content

        # --- NETTOYAGE POUR EXTRAIRE LA LISTE DIRECTEMENT ---
        # On cherche le contenu entre les crochets [ ]
        match = re.search(r'\[.*\]', raw_content, re.DOTALL)
        
        if match:
            json_str = match.group()
            # On transforme la chaîne JSON en véritable liste Python
            symptoms_list = json.loads(json_str)
            return symptoms_list
        else:
            # Si le modèle n'a pas mis de crochets, on tente de charger le texte brut
            try:
                return json.loads(raw_content)
            except:
                return {"error": "Format JSON non détecté", "raw": raw_content}

    except Exception as e:
        return {"error": "Erreur Groq Vision", "details": str(e)}

if __name__ == "__main__":
    resultat = extract_symptoms_from_image("download.jpg")
    
    # Maintenant 'resultat' est une LISTE, on peut boucler dessus
    if isinstance(resultat, list):
        print("\nSymptômes extraits avec succès :")
        for i, symptom in enumerate(resultat, 1):
            print(f"{i}. {symptom}")
    else:
        print("\nErreur ou format inattendu :", resultat)