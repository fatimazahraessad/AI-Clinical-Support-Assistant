import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Supprime la ligne : from medical_data import medical_texts <--- C'EST ELLE L'ERREUR
def generate_final_report(context, symptoms, risk_score=None): # <--- Ajoute risk_score ici
    
    # On prépare une phrase sur le score si celui-ci existe
    score_info = f"IMPORTANT: The ML model has calculated a Stroke Risk Score of {risk_score}%." if risk_score else ""

    prompt = f"""
    You are an expert Neuroradiology Assistant. Based on the following data, 
    generate a professional clinical report in ENGLISH.

    {score_info}  <--- On injecte le score ici pour que le LLM le voie

    CONTEXT FROM DATABASE:
    {context}

    EXTRACTED SYMPTOMS:
    {", ".join(symptoms)}

    REPORT STRUCTURE:
    1. SUMMARY OF ANOMALIES: Describe the findings clearly.
    2. ML RISK ANALYSIS: Interpret the calculated risk score of {risk_score}%. # <--- Ajoute cette section
    3. POTENTIAL IMPLICATIONS: Explain what these findings might mean clinically.
    4. RECOMMENDATIONS: Suggest next steps.
    5. CONCLUSION: Final synthesis.

    IMPORTANT: Maintain a professional, objective medical tone. 
    State that this is an AI-assisted analysis and requires physician validation.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content