from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_relevant_context_langchain(symptoms_list):
    # 1. Configuration du modèle d'embedding (identique au tien mais via LangChain)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connexion à ta base ChromaDB existante
    vectorstore = Chroma(
        persist_directory="./vdb", 
        embedding_function=embeddings,
        collection_name="medical_knowledge"
    )

    # 3. Création d'un "Retriever"
    # C'est un objet LangChain spécialisé dans la récupération de documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    query = " ".join(symptoms_list)
    
    # 4. Récupération des documents
    docs = retriever.invoke(query)
    
    # Extraction du contenu et des sources
    context_text = "\n".join([d.page_content for d in docs])
    sources = [d.metadata.get('source', 'ID inconnu') for d in docs]
    
    return context_text, sources