import os
import uuid
import torch
import traceback
import noisereduce as nr
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf
import speech_recognition as sr
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from emotion import predict_with_lstm
from main_imp import (
    SpeakerVerificationModel,
    train_model, verify_speaker, get_embedding, MODEL_PATH,
    SCALER_PATH, prepare_data, save_user_model, USER_MODEL_FOLDER,
    NON_USER_FOLDER, DATA_DIR
)
import pickle
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTHORIZED_USER_FOLDER = os.path.join(DATA_DIR, "authenticated_user")
TEMP_AUDIO_FOLDER = os.path.join(DATA_DIR, "temp_audio")
INPUT_SIZES = os.path.join(DATA_DIR, "input_sizes")
REGISTERED_USERS_FILE = os.path.join(DATA_DIR, 'registered_users.txt')
REGISTERED_USERS = set()

# Create necessary directories
for directory in [TEMP_AUDIO_FOLDER, AUTHORIZED_USER_FOLDER, INPUT_SIZES]:
    os.makedirs(directory, exist_ok=True)

# Global variables to store the model and scaler
model = None
scaler = None

# Helper functions remain the same

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_registered_users():
    with open(REGISTERED_USERS_FILE, 'w') as f:
        for user in REGISTERED_USERS:
            f.write(user + '\n')

def load_registered_users():
    global REGISTERED_USERS
    if os.path.exists(REGISTERED_USERS_FILE):
        with open(REGISTERED_USERS_FILE, 'r') as f:
            REGISTERED_USERS = set(f.read().splitlines())
    else:
        REGISTERED_USERS = set()

def save_registered_user(username):
    with open(REGISTERED_USERS_FILE, 'a') as f:
        f.write(username + '\n')

def denoise_audio(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        denoised_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
        sf.write(file_path, denoised_audio, sample_rate)
        print(f"Audio at {file_path} denoised successfully.")
    except Exception as e:
        print(f"Error during denoising: {str(e)}")

def detect_wake_word(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")
            return "hey buddy" in text.lower()
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return False
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return False

def load_input_size_for_user(username):
    input_size_file = os.path.join(INPUT_SIZES, "input_size.txt")
    if os.path.exists(input_size_file):
        with open(input_size_file, 'r') as f:
            for line in f:
                if line.startswith(username):
                    try:
                        return int(line.split(":")[1].strip())
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing input size for user {username}: {e}")
                        return None
    print(f"No input size found for user {username}.")
    return None

def load_model_for_user(username):
    global model, scaler
    user_model_path = os.path.join(USER_MODEL_FOLDER, username, MODEL_PATH)
    user_scaler_path = os.path.join(USER_MODEL_FOLDER, username, SCALER_PATH)
    input_size = load_input_size_for_user(username)
    
    if all(os.path.exists(p) for p in [user_model_path, user_scaler_path]) and input_size:
        try:
            model = SpeakerVerificationModel(input_size)
            model.load_state_dict(torch.load(user_model_path))
            model.eval()
            with open(user_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            print(f"Error loading model: {e}")
    return None, None

@app.get("/")
def hello():
    return JSONResponse(content={"message": "hello"})

@app.get("/ping")
def ping():
    return JSONResponse(content={"message": "pong"})
@app.get("/model_status/{username}")
def model_status(username: str):
    user_model_path = os.path.join(USER_MODEL_FOLDER, username, MODEL_PATH)
    user_scaler_path = os.path.join(USER_MODEL_FOLDER, username, SCALER_PATH)
    
    if os.path.exists(user_model_path) and os.path.exists(user_scaler_path):
        return JSONResponse(content={"message": f"Model for user {username} is already trained."})
    else:
        return JSONResponse(status_code=404, content={"error": f"No trained model found for user {username}. Please train the model first."})

@app.post("/register_user")
async def register_user(username: str = Query(...)):
    if not username:
        return JSONResponse(status_code=400, content={"error": "No username provided"})
    if username in REGISTERED_USERS:
        return JSONResponse(status_code=400, content={"error": "Username already exists"})

    REGISTERED_USERS.add(username)
    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)
    save_registered_user(username)

    return JSONResponse(content={"message": f"User {username} registered successfully"})

@app.post("/upload_sample/{username}")
async def upload_sample(username: str, audio: UploadFile = File(...)):
    if username not in REGISTERED_USERS:
        return JSONResponse(status_code=400, content={"error": "User not registered"})

    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)

    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
    
    with open(audio_path, "wb") as buffer:
        buffer.write(await audio.read())
    
    if not detect_wake_word(audio_path):
        os.remove(audio_path)
        return JSONResponse(status_code=400, content={"error": "Wake word not detected, please upload again"})

    final_audio_path = os.path.join(user_dir, unique_filename)
    os.rename(audio_path, final_audio_path)
    denoise_audio(final_audio_path)

    return JSONResponse(content={"message": f"Audio sample uploaded and wake word detected for {username}"})

@app.post("/clear_negative_samples")
def clear_negative_samples():
    if not os.path.exists(NON_USER_FOLDER):
        return JSONResponse(status_code=400, content={"error": "NON_USER_FOLDER does not exist"})
        
    for filename in os.listdir(NON_USER_FOLDER):
        file_path = os.path.join(NON_USER_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return JSONResponse(content={"message": "All negative samples have been cleared from NON_USER_FOLDER"})

@app.post("/upload_negative_sample")
async def upload_negative_sample(audio: UploadFile = File(...)):
    os.makedirs(NON_USER_FOLDER, exist_ok=True)
    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(NON_USER_FOLDER, unique_filename)

    with open(audio_path, "wb") as buffer:
        buffer.write(await audio.read())
    
    if not detect_wake_word(audio_path):
        os.remove(audio_path)
        return JSONResponse(status_code=400, content={"error": "Wake word not detected, please upload again"})

    final_audio_path = os.path.join(NON_USER_FOLDER, unique_filename)
    os.rename(audio_path, final_audio_path)
    denoise_audio(final_audio_path)

    return JSONResponse(content={"message": "Negative audio sample uploaded and wake word detected"})

@app.post("/train_model/{username}")
async def train_model_endpoint(username: str):
    global model
    model = None
    if username not in REGISTERED_USERS:
        return JSONResponse(status_code=400, content={"error": "User not registered"})

    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)
    try:
        X_train, X_test, y_train, y_test = prepare_data(user_dir, username)
        input_size = len(X_train[0]) if len(X_train) > 0 else 0
        model = train_model(username, X_train, y_train)

        if model:
            save_user_model(model, None, username, save_scaler_only=False)
            input_size_file = os.path.join(INPUT_SIZES, "input_size.txt")

            with open(input_size_file, 'a') as f:
                f.write(f"{username}: {input_size}\n")

            if os.path.exists(user_dir):  # Check if user_dir exists before accessing it
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)  # Remove the user directory after training
                print(f"User directory {user_dir} removed successfully.")
            else:
                print(f"User directory {user_dir} does not exist. Skipping cleanup.")

            return JSONResponse(content={"message": "Model trained successfully"})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "An error occurred during training."})

    return JSONResponse(status_code=500, content={"error": "Model training failed."})


@app.post("/recognize/{username}")
async def recognize_speaker_and_emotion_endpoint(username: str, audio: UploadFile = File(...)):
    if username not in REGISTERED_USERS:
        return JSONResponse(status_code=400, content={"error": "User not registered"})

    if audio.filename == '':
        return JSONResponse(status_code=400, content={"error": "No selected file"})

    create_directory(TEMP_AUDIO_FOLDER)
    user_temp_folder = os.path.join(TEMP_AUDIO_FOLDER, username)
    os.makedirs(user_temp_folder, exist_ok=True)

    unique_filename = str(uuid.uuid4()) + '.wav'
    temp_path = os.path.join(user_temp_folder, unique_filename)
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await audio.read())
    
    denoise_audio(temp_path)

    if not detect_wake_word(temp_path):
        os.remove(temp_path)
        return JSONResponse(content={"message": "Wake word not detected."}, status_code=200)

    model, scaler = load_model_for_user(username)
    if model is None or scaler is None:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": f"Model not loaded for user {username}. Please train the model first."})

    try:
        is_authorized, probability = verify_speaker(temp_path, model, scaler)
        
        if is_authorized:
            try:
                predicted_class, predicted_emotion = predict_with_lstm(temp_path)
                os.remove(temp_path)
                return JSONResponse(content={
                    "message": "Speaker identified as authorized",
                    "probability": probability,
                    "predicted_class": predicted_class.tolist(),
                    "predicted_emotion": predicted_emotion
                }, status_code=200)
            except Exception as emotion_error:
                return JSONResponse(status_code=500, content={"error": f"Error during emotion recognition: {str(emotion_error)}"})
        else:
            os.remove(temp_path)
            return JSONResponse(content={"message": "Wake word detected but speaker unidentified", "probability": probability}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error during recognition for {username}: {str(e)}"})

@app.delete("/delete_model/{username}")
async def delete_model(username: str):
    if username not in REGISTERED_USERS:
        return JSONResponse(status_code=400, content={"error": "User not registered"})

    user_model_folder = os.path.join(USER_MODEL_FOLDER, username)
    user_model_path = os.path.join(user_model_folder, MODEL_PATH)
    user_scaler_path = os.path.join(user_model_folder, SCALER_PATH)
    
    if not os.path.exists(user_model_path) and not os.path.exists(user_scaler_path):
        return JSONResponse(status_code=404, content={"message": f"No model or scaler found for user {username}."})

    try:
        if os.path.exists(user_model_path):
            os.remove(user_model_path)
            print(f"Model for user {username} deleted successfully.")

        if os.path.exists(user_scaler_path):
            os.remove(user_scaler_path)
            print(f"Scaler for user {username} deleted successfully.")

        if os.path.exists(user_model_folder) and len(os.listdir(user_model_folder)) == 0:
            os.rmdir(user_model_folder)
            print(f"User model folder {user_model_folder} deleted successfully.")

        REGISTERED_USERS.discard(username)
        save_registered_users()

        return JSONResponse(content={"message": f"Model and scaler for user {username} deleted successfully."}, status_code=200)
    except Exception as e:
        print(f"Error while deleting model and scaler for user {username}: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error while deleting model and scaler for user {username}: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    load_registered_users()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))