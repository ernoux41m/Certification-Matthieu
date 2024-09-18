import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialiser le client OpenAI avec la clé API
client = OpenAI(api_key="openai key")

# Modèle GPT à utiliser
GPT_MODEL = "gpt-4-1106-preview"  # Ou "gpt-3.5-turbo-1106"

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le format des données d'entrée
class ModelInput(BaseModel):
    prompt: str
    max_tokens: int = 100  # Limite par défaut à 100 tokens

# Utiliser bcrypt pour hasher les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Charger les valeurs sensibles depuis les variables d'environnement
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Assurez-vous que SECRET_KEY est défini
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

from passlib.context import CryptContext

# Utiliser bcrypt pour hasher les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Charger les valeurs sensibles depuis les variables d'environnement
superuser_password = os.getenv("PASSWORD_PDF")
if not superuser_password:
    raise ValueError("PASSWORD_PDF must be set in the environment.")

fake_users_db = {
    os.getenv("SUPERUSER_USERNAME"): {
        "username": os.getenv("SUPERUSER_USERNAME"),
        "full_name": "Admin User",
        "hashed_password": pwd_context.hash(superuser_password),  # Hash the password here
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Dépendance pour vérifier le token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    print(f"Plain password: {password}")  # Affichez le mot de passe en clair
    print(f"Hashed password: {user['hashed_password']}")  # Affichez le mot de passe haché
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data["username"])
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if current_user["disabled"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Endpoint pour obtenir un token d'accès
@app.post("/token", response_model=dict)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Utiliser OpenAI GPT pour générer du texte avec le client OpenAI
@app.post("/generate/", dependencies=[Depends(get_current_active_user)])
def generate_text(input_data: ModelInput):
    try:
        # Préparer les messages pour la requête
        messages = [
            {"role": "system", "content": "You answer questions about Web services."},
            {"role": "user", "content": input_data.prompt}
        ]
        
        # Envoyer la requête à OpenAI
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0
        )
        
        # Extraire le contenu du message de la réponse
        response_message = response.choices[0].message.content
        
        return {"output": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint racine pour tester que l'API fonctionne
@app.get("/")
def read_root():
    return {"message": "Welcome to GPT API!"}
