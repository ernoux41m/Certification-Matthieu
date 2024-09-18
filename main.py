import streamlit as st
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from dotenv import load_dotenv
import os
from openai import OpenAI

# Charger les variables d'environnement (comme la clé API OpenAI)
load_dotenv()

# Initialiser le client OpenAI avec la clé API
client = OpenAI(api_key="Open-ai")

def get_access_token(username, password):
    url = "http://127.0.0.1:8000/token"
    payload = {
        "username": username,
        "password": password
    }
    
    # Envoyer les données sous forme de formulaire
    response = requests.post(url, data=payload)  # 'data' est utilisé pour les formulaires
    
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        if access_token:
            return access_token
        else:
            print("Pas de jeton dans la réponse.")
            return None
    else:
        print(f"Erreur lors de l'authentification : {response.status_code} - {response.text}")
        return None



# Fonction pour obtenir les embeddings via OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")  # Remplacer les retours à la ligne par des espaces
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Fonction pour créer les embeddings en utilisant OpenAI
def create_embeddings(text):
    paragraphs = text.split("\n\n")  # Diviser le texte en paragraphes
    
    # Créer une liste pour stocker les embeddings de chaque paragraphe
    embeddings = [get_embedding(paragraph) for paragraph in paragraphs]
    
    return paragraphs, np.array(embeddings)

# Fonction pour trouver le texte le plus pertinent dans le PDF
def find_most_relevant_text(question, paragraphs, embeddings):
    # Générer l'embedding de la question
    question_embedding = np.array(get_embedding(question)).reshape(1, -1)
    
    # Calculer la similarité cosinus entre la question et les paragraphes
    similarities = cosine_similarity(question_embedding, embeddings)
    most_relevant_index = np.argmax(similarities)
    return paragraphs[most_relevant_index]

def query_llm_api(prompt, context):
    # Limiter la taille du contexte à quelques centaines de tokens
    if len(context) > 2000:  # Vous pouvez ajuster cette limite en fonction de votre besoin
        context = context[:2000]  # Troncature du texte pour rester dans la limite

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
    headers = {
        "Authorization": f"Bearer {st.session_state.access_token}"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("output")
    else:
        return f"Erreur lors de l'appel à l'API LLM: {response.status_code} - {response.text}"


# Obtenir les informations d'identification à partir des variables d'environnement
USERNAME_PDF = os.getenv("USERNAME_PDF")
PASSWORD = os.getenv("PASSWORD")

# Fonction pour vérifier les informations de connexion
def check_credentials(username, password):
    # Cette fonction peut vérifier les informations d'identification via une API ou comparer localement
    return username == USERNAME_PDF and password == PASSWORD

# Interface Streamlit
st.title("PDF Reader")

# Initialiser l'état de la session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "access_token" not in st.session_state:  # Initialiser le token s'il n'existe pas
    st.session_state.access_token = None
    
if not st.session_state.logged_in:
    st.subheader("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        # Récupérer et stocker le token directement dans st.session_state
        access_token = get_access_token(username, password)
        if access_token:
            st.session_state.access_token = access_token
            st.session_state.logged_in = True
            st.success("Connexion réussie !")
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

else:
    # Interface après connexion réussie
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
            # Affichage de la conversation avec les boutons "Like" et "Dislike"
            # Affichage de la conversation avec les boutons "Like" et "Dislike"
            # Affichage de la conversation avec les boutons "Like" et "Dislike"
            # Affichage de la conversation avec les boutons "Like" et "Dislike"
            # Affichage de la conversation avec les boutons "Like" et "Dislike"
            for index, entry in enumerate(st.session_state.history):
                if entry["role"] == "user":
                    st.write(f"**Vous**: {entry['content']}")
                elif entry["role"] == "llm":
                    st.write(f"**LLM**: {entry['content']}")

                    # Initialisation de l'état pour les boutons Like et Dislike s'ils n'existent pas encore
                    if f"like_{index}" not in st.session_state:
                        st.session_state[f"like_{index}"] = False
                    if f"dislike_{index}" not in st.session_state:
                        st.session_state[f"dislike_{index}"] = False

                    # Ajout des boutons "Like" et "Dislike" pour la réponse LLM
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Like", key=f"like_btn_{index}"):
                            st.session_state[f"like_{index}"] = True
                            st.session_state[f"dislike_{index}"] = False  # Désactiver Dislike si Like est cliqué
                    with col2:
                        if st.button("👎 Dislike", key=f"dislike_btn_{index}"):
                            st.session_state[f"dislike_{index}"] = True
                            st.session_state[f"like_{index}"] = False  # Désactiver Like si Dislike est cliqué

                    # Affichage d'un message de feedback si l'utilisateur a cliqué sur un bouton
                    if st.session_state[f"like_{index}"]:
                        st.success("Vous avez aimé cette réponse.")
                    elif st.session_state[f"dislike_{index}"]:
                        st.error("Vous n'avez pas aimé cette réponse.")






    # Bouton de déconnexion
    if st.button("Se déconnecter"):
        st.session_state.logged_in = False
        st.session_state.history = []
        st.write("Déconnecté. Veuillez rafraîchir la page pour vous reconnecter.")
