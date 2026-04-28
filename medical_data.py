import chromadb
from datasets import load_dataset
from chromadb.utils import embedding_functions
import os

def initialize_vector_db():
    # 1. Initialisation du client (crée le dossier vdb sur le disque)
    chroma_client = chromadb.PersistentClient(path="./vdb")
    
    # 2. Modèle d'embedding open-source
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 3. Création ou récupération de la collection
    collection = chroma_client.get_or_create_collection(name="medical_knowledge", embedding_function=ef)

    print("--- Début du processus d'ingestion ---")
    
    # 4. Chargement du dataset depuis Hugging Face
    print("Téléchargement du dataset ruslanmv/ai-medical-dataset...")
    # On charge une plus grande partie pour pouvoir filtrer les cas intéressants
    ds = load_dataset("ruslanmv/ai-medical-dataset", split="train[:5000]") 
    
    # 5. Filtrage : on ne garde que les cas liés à la neurologie/IRM
    keywords = ['mri', 'brain', 'headache', 'neurology', 'cerebral', 'lesion', 'stroke']
    filtered_entries = [
        entry for entry in ds 
        if any(word in entry['Description'].lower() for word in keywords)
    ]
    
    print(f"Filtrage terminé : {len(filtered_entries)} cas pertinents trouvés.")

    # 6. Ajout à ChromaDB
    print("Indexation dans la base vectorielle...")
    for i, entry in enumerate(filtered_entries):
        content = f"Case: {entry['Description']} \nAdvice: {entry['Doctor']}"
        collection.add(
            documents=[content],
            ids=[f"hf_neuro_{i}"],
            metadatas=[{"source": "huggingface", "specialty": "neurology"}]
        )
    
    print(f"--- Base de données prête ! ({collection.count()} documents indexés) ---")

# --- CETTE PARTIE EST ESSENTIELLE POUR LANCER LE SCRIPT ---
if __name__ == "__main__":
    initialize_vector_db()