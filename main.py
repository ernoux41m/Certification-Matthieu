import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Charger le modèle d'embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Fonction pour créer les embeddings du texte
def create_embeddings(text):
    paragraphs = text.split("\n\n")  # Diviser le texte en paragraphes
    embeddings = embedder.encode(paragraphs)
    return paragraphs, embeddings

# Fonction pour trouver le texte le plus pertinent dans le PDF
def find_most_relevant_text(question, paragraphs, embeddings):
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)
    most_relevant_index = np.argmax(similarities)
    return paragraphs[most_relevant_index]

# Fonction pour interroger l'API LLM avec un contexte
def query_llm_api(prompt, context):
    llm_prompt = (
        f"Voici le contexte extrait d'un document :\n\n"
        f"{context}\n\n"
        f"Question : {prompt}\n"
        f"Réponds uniquement en te basant sur le texte ci-dessus."
    )
    url = "http://127.0.0.1:8000/generate/"
    payload = {
        "prompt": llm_prompt,
        "max_tokens": 200
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("output")
    else:
        return "Erreur lors de l'appel à l'API LLM"

# Interface Streamlit
st.title("LLM PDF QA Chatbot")
st.write("Télécharge un PDF et pose des questions basées uniquement sur son contenu.")

# Télécharger le PDF
uploaded_file = st.file_uploader("Télécharger un fichier PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extraction du texte du PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        paragraphs, embeddings = create_embeddings(pdf_text)
        st.success("Texte extrait et embeddings créés avec succès !")

    # Conversation avec l'utilisateur
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Posez une question:")

    if user_input:
        relevant_text = find_most_relevant_text(user_input, paragraphs, embeddings)
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "system", "content": relevant_text})

        # Appel à l'API avec le contexte trouvé
        llm_response = query_llm_api(user_input, relevant_text)
        st.session_state.history.append({"role": "llm", "content": llm_response})

        # Affichage de la conversation
        for entry in st.session_state.history:
            if entry["role"] == "user":
                st.write(f"**Vous**: {entry['content']}")
            elif entry["role"] == "llm":
                st.write(f"**LLM**: {entry['content']}")

