# AI-Clinical-Support-Assistant (MedAI) 🏥

**MedAI** est une plateforme d'aide au diagnostic clinique spécialisée dans la détection précoce des accidents vasculaires cérébraux (AVC). Ce projet utilise une architecture hybride combinant le Machine Learning classique et l'Intelligence Artificielle Générative (RAG).

##  Fonctionnalités
- **Extraction Multimodale :** Analyse des symptômes à partir de texte, de fichiers audio (transcription via Whisper) et d'images.
- **Modèle MLP (Multi-Layer Perceptron) :** Calcul statistique de la probabilité de risque d'AVC basé sur des facteurs de santé (âge, hypertension, glucose, IMC, tabagisme).
- **Pipeline RAG (Retrieval-Augmented Generation) :** Génération d'un rapport clinique détaillé en interrogeant une base de connaissances médicale via LangChain et Groq (Llama 3).
- **Analyse Bilingue :** Support de l'extraction des données en français et en anglais.

##  Architecture Technique
Le système repose sur deux briques majeures :
1. **Predictive Engine :** Un réseau de neurones MLP entraîné sur le dataset `healthcare-stroke-data`.
2. **Analysis Engine :** Un moteur RAG utilisant ChromaDB (Vector Store) pour fournir un contexte médical précis au LLM.
