import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from speechbrain.inference import EncoderClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import torch.nn.functional as F
import shutil

# Constants
DATA_DIR = "data"
USER_MODEL_FOLDER = os.path.join(DATA_DIR, "UserModel")
USER_FOLDER = os.path.join(DATA_DIR, "authenticated_user")
NON_USER_FOLDER = "non_target"
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
ECAPA_TDNN_EMBEDDING_SIZE = 192

# Load the pre-trained model for embedding extraction
def load_pretrained_encoder():
    pretrained_model_path = "/root/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb"
    
    # Check if the model already exists
    if os.path.exists(pretrained_model_path):
        print("Using cached model.")
    else:
        print("Cached model not found. Downloading the model.")
    
    # Load the pretrained model (will use the cache if available)
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return encoder

encoder = load_pretrained_encoder()

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        weights = F.softmax(self.attention(x), dim=1)
        weighted_sum = torch.sum(weights * x, dim=1)
        # Normalize the pooled embeddings
        normalized_embedding = self.layer_norm(weighted_sum)
        return normalized_embedding

class SpeakerVerificationModel(nn.Module):
    def __init__(self, input_size=ECAPA_TDNN_EMBEDDING_SIZE):
        super(SpeakerVerificationModel, self).__init__()
        self.attention_pooling = AttentionPooling(input_size)
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.attention_pooling(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.sigmoid(self.fc2(x))
        return x

def get_embedding(audio_path):
    """Get speaker embedding from the audio."""
    try:
        signal, fs = librosa.load(audio_path, sr=16000)
        signal = torch.tensor(np.expand_dims(signal, axis=0))
        embeddings = encoder.encode_batch(signal)
        return embeddings.squeeze().cpu().detach().numpy()
    except Exception as e:
        print(f"Error in getting embedding for {audio_path}: {e}")
        return None

def collect_embeddings(folder, label):
    """Collect embeddings from audio files in the given folder."""
    embeddings = []
    labels = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            embedding = get_embedding(file_path)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label)
    return embeddings, labels

def prepare_data(user_dir, username):
    """Prepare data for training the model."""
    user_embeddings, user_labels = collect_embeddings(user_dir, 1)
    non_user_embeddings, non_user_labels = collect_embeddings(NON_USER_FOLDER, 0)
    
    X = user_embeddings + non_user_embeddings
    y = user_labels + non_user_labels
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use in the specific user's folder
    save_user_model(None, scaler, username, save_scaler_only=True)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(username, X_train, y_train):
    print(f"Preparing data for user {username}...")

    input_size = X_train[0].shape[0]
    model = SpeakerVerificationModel(input_size)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        max_grad_norm = 2.0
        
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = torch.tensor(X_train[i:i+BATCH_SIZE], dtype=torch.float32).unsqueeze(1).to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_y = torch.tensor(y_train[i:i+BATCH_SIZE], dtype=torch.float32).unsqueeze(1).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        
        if (epoch + 1) % 10 == 0:
            epoch_loss = total_loss / len(X_train)
            epoch_accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training completed.")
    
    # Save the model and scaler
    save_user_model(model, None, username, save_scaler_only=False)
    
    # Clean up the user's folder after training
    remove_user_folder(username)
    
    return model


def verify_speaker(audio_path, model, scaler):
    """Verify if the audio is from the authorized user."""
    embedding = get_embedding(audio_path)
    if embedding is None:
        return False, 0.0

    embedding_scaled = scaler.transform([embedding])
    
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(embedding_scaled, dtype=torch.float32).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
        probability = output.item()
    return probability > 0.75, probability


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


# Folder Management Functions (no changes required here)
def create_user_folder(username):
    user_folder_path = os.path.join(USER_FOLDER, username)
    os.makedirs(user_folder_path, exist_ok=True)
    return user_folder_path

def create_user_model_folder(username):
    user_model_folder_path = os.path.join(USER_MODEL_FOLDER, username)
    os.makedirs(user_model_folder_path, exist_ok=True)
    return user_model_folder_path

def remove_user_folder(username):
    user_folder_path = os.path.join(USER_FOLDER, username)
    if os.path.exists(user_folder_path):
        for file_name in os.listdir(user_folder_path):
            file_path = os.path.join(user_folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(user_folder_path)
        print(f"User folder {user_folder_path} removed successfully.")

def save_user_model(model, scaler, username, save_scaler_only=False):
    # Define the path for the user-specific folder inside the 'UserModel' directory
    user_model_folder = os.path.join(USER_MODEL_FOLDER, username)
    create_directory(user_model_folder)
    
    # Create the directory if it doesn't exist
    os.makedirs(user_model_folder, exist_ok=True)
    
    # Save the model if provided and not set to save only the scaler
    if not save_scaler_only and model is not None:
        model_path = os.path.join(user_model_folder, MODEL_PATH)
        # Save only the state dict of the model
        torch.save(model.state_dict(), model_path)  # Save the state dictionary
        print(f"Model saved at {model_path}")
    
    # Save the scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(user_model_folder, SCALER_PATH)
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        print(f"Scaler saved at {scaler_path}")
