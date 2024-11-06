import os
import uuid
import torch
import traceback
import noisereduce as nr
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS
from IPython.display import Audio, display
from emotion import predict_with_lstm
from main_imp import (
    SpeakerVerificationModel,
    train_model, verify_speaker, get_embedding, MODEL_PATH,
    SCALER_PATH, prepare_data, save_user_model, USER_MODEL_FOLDER,
    SCALER_PATH, MODEL_PATH, NON_USER_FOLDER, DATA_DIR)
import pickle

app = Flask(__name__)
CORS(app)


AUTHORIZED_USER_FOLDER = os.path.join(DATA_DIR, "authenticated_user")
TEMP_AUDIO_FOLDER = os.path.join(DATA_DIR, "temp_audio")
INPUT_SIZES = os.path.join(DATA_DIR, "input_sizes")
REGISTERED_USERS_FILE =os.path.join(DATA_DIR,'registered_users.txt')
REGISTERED_USERS = set() 

# Create necessary directories
for directory in [TEMP_AUDIO_FOLDER, AUTHORIZED_USER_FOLDER, INPUT_SIZES]:
    os.makedirs(directory, exist_ok=True)
# Global variables to store the model and scaler
model = None
scaler = None

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_registered_users():
    """Save registered users to file."""
    with open(REGISTERED_USERS_FILE, 'w') as f:
        for user in REGISTERED_USERS:
            f.write(user + '\n')

def load_registered_users():
    """Load registered users from file."""
    global REGISTERED_USERS
    if os.path.exists(REGISTERED_USERS_FILE):
        with open(REGISTERED_USERS_FILE, 'r') as f:
            REGISTERED_USERS = set(f.read().splitlines())
    else:
        REGISTERED_USERS = set()

def save_registered_user(username):
    """Save a single registered user to file."""
    with open(REGISTERED_USERS_FILE, 'a') as f:
        f.write(username + '\n')

def denoise_audio(file_path):
    """Denoise audio file at given path."""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        denoised_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
        sf.write(file_path, denoised_audio, sample_rate)
        print(f"Audio at {file_path} denoised successfully.")
    except Exception as e:
        print(f"Error during denoising: {str(e)}")

def detect_wake_word(audio_file_path):
    """Detect wake word in audio file."""
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
    """Load input size for specific user."""
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
    """Load model and scaler for specific user."""
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

# Flask routes
@app.route('/', methods=['GET'])
def hello():
    """Root endpoint."""
    return "hello", 200

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    return jsonify({'message': 'pong'}), 200

@app.route('/model_status/<username>', methods=['GET'])
def model_status(username):
    user_model_path = os.path.join(USER_MODEL_FOLDER, username, MODEL_PATH)
    user_scaler_path = os.path.join(USER_MODEL_FOLDER, username, SCALER_PATH)
    
    if os.path.exists(user_model_path) and os.path.exists(user_scaler_path):
        return jsonify({'message': f'Model for user {username} is already trained.'}), 200
    else:
        return jsonify({'message': f'No trained model found for user {username}. Please train the model first.'}), 404




@app.route('/register_user', methods=['POST'])
def register_user():
    username = request.json.get('username')
    if not username:
        return jsonify({'error': 'No username provided'}), 400
    if username in REGISTERED_USERS:
        return jsonify({'error': 'Username already exists'}), 400

    REGISTERED_USERS.add(username)
    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)  # Create a directory for the user

    # Save the username to the registered users file
    save_registered_user(username)

    return jsonify({'message': f'User {username} registered successfully'}), 200


@app.route('/upload_sample/<username>', methods=['POST'])
def upload_sample(username):
    # Check if the user is registered
    if username not in REGISTERED_USERS:
        return jsonify({'error': 'User not registered'}), 400
    create_directory(AUTHORIZED_USER_FOLDER)
    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)  # Create a user-specific folder if it doesn't exist

    # Check if the file part is present in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Ensure a file has been selected
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the audio file already exists
    existing_files = os.listdir(user_dir)
    if audio_file.filename in existing_files:
        return jsonify({'message': f'Sample {audio_file.filename} already uploaded.'}), 200

    # Save the audio file with a unique filename
    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
    
    try:
        audio_file.save(audio_path)
        print(f"Audio sample saved at {audio_path}")

        # Check for the wake word in the audio file
        if not detect_wake_word(audio_path):
            os.remove(audio_path)  # Remove temporary file if wake word is not detected
            return jsonify({'message': 'Wake word not detected, please upload again'}), 400

        # Move the processed audio to the user's directory
        final_audio_path = os.path.join(user_dir, unique_filename)
        os.rename(audio_path, final_audio_path)
        denoise_audio(final_audio_path)

        return jsonify({'message': f'Audio sample uploaded and wake word detected for {username}'}), 200
    except Exception as e:
        print(f"Error while processing audio upload for {username}: {e}")
        # Cleanup the temporary file in case of an error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': f'Error while uploading audio: {str(e)}'}), 500



@app.route('/clear_negative_samples', methods=['POST'])
def clear_negative_samples():
    try:
        # Ensure that the NON_USER_FOLDER exists
        if not os.path.exists(NON_USER_FOLDER):
            return jsonify({'error': 'NON_USER_FOLDER does not exist'}), 400
        
        # Remove all files in the NON_USER_FOLDER
        for filename in os.listdir(NON_USER_FOLDER):
            file_path = os.path.join(NON_USER_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({'message': 'All negative samples have been cleared from NON_USER_FOLDER'}), 200
    except Exception as e:
        print(f"Error while clearing NON_USER_FOLDER: {e}")
        return jsonify({'error': f'Error while clearing folder: {str(e)}'}), 500



@app.route('/upload_negative_sample', methods=['POST'])
def upload_negative_sample():
    # Ensure that the NON_USER_FOLDER exists
    os.makedirs(NON_USER_FOLDER, exist_ok=True)

    # Check if the file part is present in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Ensure a file has been selected
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the audio file with a unique filename
    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(NON_USER_FOLDER, unique_filename)

    try:
        audio_file.save(audio_path)
        print(f"Negative audio sample saved at {audio_path}")

        # Check for the wake word in the audio file
        if not detect_wake_word(audio_path):
            os.remove(audio_path)  # Remove the file if wake word is not detected
            return jsonify({'message': 'Wake word not detected, please upload again'}), 400

        # Move the processed audio to the NON_USER_FOLDER
        final_audio_path = os.path.join(NON_USER_FOLDER, unique_filename)
        os.rename(audio_path, final_audio_path)
        denoise_audio(final_audio_path)

        return jsonify({'message': 'Negative audio sample uploaded and wake word detected'}), 200
    except Exception as e:
        print(f"Error while processing negative audio sample: {e}")
        # Cleanup the temporary file in case of an error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': f'Error while uploading audio: {str(e)}'}), 500


@app.route('/train_model/<username>', methods=['POST'])
def train_model_endpoint(username):
    global model
    model= None

    print("Checking registered users...")  # Log message
    print(f"Registered users (set): {REGISTERED_USERS}")
    user_dir = os.path.join(AUTHORIZED_USER_FOLDER, username)

    # Print each registered user
    for user in REGISTERED_USERS:
        print(f"Registered user: {user}")

    # Check if the user is registered
    if username not in REGISTERED_USERS:
        print(f"User {username} not registered.")  # Log message if user is not found
        return jsonify({'error': 'User not registered'}), 400

    try:
        # Check if a model for the user already exists
        if model is not None:
            return jsonify({'message': f'Model for user {username} is already trained. Please retrain if necessary.'}), 200

        print(f"Starting model training for {username}...")

        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_data(user_dir, username)  # Pass the username to prepare_data

        # Determine the input size (assuming it's the same for all users)
        input_size = len(X_train[0]) if len(X_train) > 0 else 0

        # Train the model
        model = train_model(username, X_train, y_train)  # Pass the username and training data

        if model:
            # Save the model and scaler using the save_user_model function
            save_user_model(model, None, username, save_scaler_only=False)
            print(f"Model and scaler saved for {username}.")

            # Create input size folder and file
            input_size_folder = INPUT_SIZES
            os.makedirs(input_size_folder, exist_ok=True)
            input_size_file = os.path.join(input_size_folder, "input_size.txt")

            # Write username and input size to the file
            with open(input_size_file, 'a') as f:
                f.write(f"{username}: {input_size}\n")
                print(f"Completed appending input size for the {username}\n")

            # Clean up the user's audio files from the `authenticated_user` folder to avoid keeping unnecessary files
            if os.path.exists(user_dir):  # Check if user_dir exists before accessing it
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)  # Remove the user directory after training
                print(f"User directory {user_dir} removed successfully.")
            else:
                print(f"User directory {user_dir} does not exist. Skipping cleanup.")

            return jsonify({'message': f'Training completed successfully for {username}!'}), 200
        else:
            return jsonify({'message': f'Training failed for {username}.'}), 500
    except Exception as e:
        print(f"Error during training for {username}: {e}")
        return jsonify({'error': f'Error during training for {username}: {str(e)}'}), 500

@app.route('/recognize/<username>', methods=['POST'])
def recognize_speaker_and_emotion_endpoint(username):
    if username not in REGISTERED_USERS:
        return jsonify({'error': 'User not registered'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    create_directory(TEMP_AUDIO_FOLDER)
    # Create a user-specific temp folder
    user_temp_folder = os.path.join(TEMP_AUDIO_FOLDER, username)
    os.makedirs(user_temp_folder, exist_ok=True)

    # Generate a unique filename for the audio file
    unique_filename = str(uuid.uuid4()) + '.wav'
    temp_path = os.path.join(user_temp_folder, unique_filename)
    audio_file.save(temp_path)
    
    # Denoise the audio file
    denoise_audio(temp_path)

    # Step 1: Detect wake word
    if not detect_wake_word(temp_path):
        os.remove(temp_path)
        return jsonify({'message': 'Wake word not detected.'}), 200

    # Step 2: Load the model and scaler for the specific user
    model, scaler = load_model_for_user(username)
    if model is None or scaler is None:
        os.remove(temp_path)
        return jsonify({'error': f'Model not loaded for user {username}. Please train the model first.'}), 500

    try:
        # Step 3: Verify the speaker
        is_authorized, probability = verify_speaker(temp_path, model, scaler)
        
        if is_authorized:
            # Step 4: If speaker is verified, perform emotion recognition
            try:
                predicted_class, predicted_emotion = predict_with_lstm(temp_path)
                os.remove(temp_path)  # Clean up the audio file after processing
                return jsonify({
                    'message': 'Speaker identified as authorized',
                    'probability': probability,
                    'predicted_class': predicted_class.tolist(),
                    'predicted_emotion': predicted_emotion
                }), 200
            except Exception as emotion_error:
                return jsonify({'error': f'Error during emotion recognition: {str(emotion_error)}'}), 500
        else:
            os.remove(temp_path)  # Clean up the audio file after processing
            return jsonify({'message': 'Wake word detected but speaker unidentified', 'probability': probability}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error during recognition for {username}: {str(e)}'}), 500


@app.route('/delete_model/<username>', methods=['DELETE'])
def delete_model(username):
    # Check if the user is registered
    if username not in REGISTERED_USERS:
        return jsonify({'error': 'User not registered'}), 400

    # Define paths for the model, scaler, and user model folder
    user_model_folder = os.path.join(USER_MODEL_FOLDER, username)
    user_model_path = os.path.join(user_model_folder, MODEL_PATH)
    user_scaler_path = os.path.join(user_model_folder, SCALER_PATH)
    
    # Check if the model and scaler exist
    if not os.path.exists(user_model_path) and not os.path.exists(user_scaler_path):
        return jsonify({'message': f'No model or scaler found for user {username}.'}), 404

    try:
        # Remove the model and scaler files
        if os.path.exists(user_model_path):
            os.remove(user_model_path)
            print(f"Model for user {username} deleted successfully.")

        if os.path.exists(user_scaler_path):
            os.remove(user_scaler_path)
            print(f"Scaler for user {username} deleted successfully.")

        # Remove the user model folder if empty
        if os.path.exists(user_model_folder) and len(os.listdir(user_model_folder)) == 0:
            os.rmdir(user_model_folder)
            print(f"User model folder {user_model_folder} deleted successfully.")

        # Optionally, remove the user from the registered users file
        REGISTERED_USERS.discard(username)
        save_registered_users()

        return jsonify({'message': f'Model and scaler for user {username} deleted successfully.'}), 200
    except Exception as e:
        print(f"Error while deleting model and scaler for user {username}: {str(e)}")
        return jsonify({'error': f'Error while deleting model and scaler for user {username}: {str(e)}'}), 500

if __name__ == '__main__':
    # Load registered users from file at startup
    load_registered_users()
    print(f"Registered users: {REGISTERED_USERS}")
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
