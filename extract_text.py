from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client_text = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN
)

def extract_symptoms_from_text(text):
    messages = [
        {
            "role": "system",
            "content": """You are a clinical NER extractor. Your only task is to extract SYMPTOMS mentioned in the text.

Rules:
- Extract symptoms only.
- Ignore diagnoses, medications, test results, medical history, treatments, and recommendations.
- No full sentences, no explanations.
- Output must be a valid JSON array of strings.

Output format:
[
  "symptom1",
  "symptom2",
  "symptom3"
]"""
        },
        {"role": "user", "content": text}
    ]

    response = client_text.chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0
    )

    content = response.choices[0].message["content"]

    # Extract JSON from markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        content = json_match.group(1)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Return raw string if parsing fails
        return content