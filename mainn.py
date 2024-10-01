import os
import uuid
import torch
import traceback
import noisereduce as nr
import librosa
import soundfile as sf
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS
from gtts import gTTS
from IPython.display import Audio, display
from imp import VariableLengthSpeakerVerificationModel, train_model, verify_speaker, get_embedding, MODEL_PATH, SCALER_PATH

app = Flask(__name__)
CORS(app)

AUTHORIZED_USER_FOLDER = "authenticated_user"
TEMP_AUDIO_FOLDER = "temp_audio"

# Create necessary directories
os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
os.makedirs(AUTHORIZED_USER_FOLDER, exist_ok=True)

# Global variables to store the model and scaler
model = None
scaler = None

# Function to denoise audio
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

    # Load the audio file and apply the recognizer
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            # Transcribe the audio to text
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")

            # Check if the recognized text contains "hi siri"
            if text.lower() == "hi buddy":
                return True
            else:
                print("Wake word not detected.")
                return False
        except sr.UnknownValueError:
            # Handle cases where the speech is unclear or not understandable
            print("Could not understand the audio")
            return False
        except sr.RequestError as e:
            # Handle any issues with the speech recognition service
            print(f"Speech recognition error: {e}")
            return False


# Function to generate and play TTS response (for feedback in Colab)
def generate_tts_response(text, output_file):
    tts = gTTS(text=text)
    tts.save(output_file)
    display(Audio(output_file, autoplay=True))

# Function to load the model and scaler once at server startup
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing speaker verification model...")
        input_size = len(get_embedding(os.path.join(AUTHORIZED_USER_FOLDER, os.listdir(AUTHORIZED_USER_FOLDER)[0])))
        model = VariableLengthSpeakerVerificationModel(input_size)
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            scaler = torch.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading model: {e}. You may need to retrain the model.")
            model = None
            scaler = None
    else:
        print("Model or scaler not found. Please train the model first.")


@app.route('/', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200



@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

@app.route('/model_status', methods=['GET'])
def model_status():
    if model is not None and scaler is not None:
        return jsonify({'message': 'Model is already trained and loaded.'}), 200
    else:
        return jsonify({'message': 'No trained model found. Please train the model first.'}), 404

# Modify the /upload_sample route to check for wake word detection before accepting the sample
@app.route('/upload_sample', methods=['POST'])
def upload_sample():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
    audio_file.save(audio_path)
    print(f"Audio sample saved at {audio_path}")

    # Check for the wake word before proceeding
    if not detect_wake_word(audio_path):
        os.remove(audio_path)  # Delete the sample if wake word is not detected
        generate_tts_response("Wake word not detected, please try again.", "retry.mp3")
        return jsonify({'message': 'Wake word not detected, please upload the sample again'}), 400

    # Move the valid sample to the authenticated_user folder
    final_audio_path = os.path.join(AUTHORIZED_USER_FOLDER, unique_filename)
    os.rename(audio_path, final_audio_path)
    print(f"Audio sample moved to {final_audio_path}")

    # Apply denoiser to the uploaded audio
    denoise_audio(final_audio_path)

    return jsonify({'message': 'Audio sample uploaded and wake word detected'}), 200


@app.route('/train_model', methods=['GET', 'POST'])
def train_model_endpoint():
    global model
    try:
        if model is not None:  # Check if the model is already trained
            return jsonify({'message': 'Model is already trained. Please retrain if necessary.'}), 200
        
        print("Starting model training...")
        model = train_model()
        if model:
            load_model()  # Reload the model and scaler after training
            return jsonify({'message': 'Training completed successfully!'}), 200
        else:
            return jsonify({'message': 'Training failed.'}), 500
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': f'Error during training: {str(e)}'}), 500



@app.route('/recognize', methods=['POST'])
def recognize_speaker_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400  # Only return 400 for no file

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Only return 400 for empty file

    # Save the uploaded audio file
    unique_filename = str(uuid.uuid4()) + '.wav'
    temp_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
    audio_file.save(temp_path)
    print(f"Audio file saved at {temp_path}")

    # Apply denoising (optional) if wake word was detected
    denoise_audio(temp_path)


    # Check for the wake word first
    if not detect_wake_word(temp_path):
        # Case 1: Wake word not detected
        os.remove(temp_path)  # Clean up temp file
        return jsonify({'message': 'Wake word not detected.'}), 200  # Always return 200 for a valid request, even if the wake word is not found

    
    if model is None or scaler is None:  # Check if the model is loaded
        print("Model or scaler is not loaded, cannot proceed with recognition.")
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500  # Use 500 if internal model issue

    try:
        print(f"Processing audio file for recognition: {temp_path}")
        is_authorized, probability = verify_speaker(temp_path, model, scaler)

        # Clean up temporary audio file
        os.remove(temp_path)
        print(f"Temporary file {temp_path} deleted.")

        if is_authorized:
            # Case 3: Speaker verified as authorized
            return jsonify({'message': 'Speaker identified as authorized', 'probability': probability}), 200
        else:
            # Case 2: Wake word detected but speaker not identified
            return jsonify({'message': 'Wake word detected but speaker unidentified', 'probability': probability}), 200
    except Exception as e:
        print(f"Error during recognition: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error during recognition: {str(e)}'}), 500




# Load model once at startup
load_model()

# Disable auto-reloading to avoid unnecessary restarts
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
