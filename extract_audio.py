import os
from dotenv import load_dotenv
from groq import Groq
from extract_text import extract_symptoms_from_text 

load_dotenv()

# Récupère ta clé API Groq (Va sur https://console.groq.com/ pour en créer une gratuite)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def extract_symptoms_from_audio(audio_path):
    try:
        if not os.path.exists(audio_path):
            return {"error": f"Fichier introuvable : {audio_path}"}

        print(f"--- Transcription en cours avec Groq (Ultra Rapide) ---")
        
        # 1. Envoi à Groq
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                prompt="Consultation médicale, Darija, symptômes, fievre, mrid.", # Ton prompt ici
                response_format="json",
                language="fr" # Ou laisse None pour auto-détection (Darija/Fr)
            )

        transcription_text = transcription.text
        print(f"Texte extrait : {transcription_text}")

        # 2. Extraction des symptômes
        if not transcription_text:
            return {"error": "Aucun texte extrait."}

        symptoms = extract_symptoms_from_text(transcription_text)

        return {
            "success": True,
            "transcription": transcription_text,
            "symptoms": symptoms
        }

    except Exception as e:
        return {"error": "Erreur Groq", "details": str(e)}

if __name__ == "__main__":
    # Test
    res = extract_symptoms_from_audio("audios/test_darija.mp3")
    print(res)