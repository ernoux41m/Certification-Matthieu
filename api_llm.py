from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gpt4all import GPT4All

# Charger le modèle GPT4All
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le format des données d'entrée
class ModelInput(BaseModel):
    prompt: str
    max_tokens: int = 100  # Limite par défaut à 100 tokens

# Endpoint pour générer du texte
@app.post("/generate/")
def generate_text(input_data: ModelInput):
    try:
        output = model.generate(input_data.prompt, max_tokens=input_data.max_tokens)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exemple d'endpoint racine
@app.get("/")
def read_root():
    return {"message": "Welcome to GPT4All API!"}
