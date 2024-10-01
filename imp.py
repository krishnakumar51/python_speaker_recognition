import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from speechbrain.pretrained import EncoderClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
USER_FOLDER = "authenticated_user"
NON_USER_FOLDER = "non_target"
MODEL_PATH = "speaker_verification_model.pth"
SCALER_PATH = "scaler.pkl"

# Load the pre-trained model for embedding extraction
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

class VariableLengthSpeakerVerificationModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def get_embedding(audio_path):
    """Get speaker embedding from the audio."""
    signal, fs = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(np.expand_dims(signal, axis=0))
    embeddings = encoder.encode_batch(signal)
    return embeddings.squeeze().cpu().detach().numpy()

def collect_embeddings(folder, label):
    """Collect embeddings from audio files in the given folder."""
    embeddings = []
    labels = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            embedding = get_embedding(file_path)
            embeddings.append(embedding)
            labels.append(label)
    return embeddings, labels

def prepare_data():
    """Prepare data for training the model."""
    user_embeddings, user_labels = collect_embeddings(USER_FOLDER, 1)
    non_user_embeddings, non_user_labels = collect_embeddings(NON_USER_FOLDER, 0)
    
    X = user_embeddings + non_user_embeddings
    y = user_labels + non_user_labels
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    torch.save(scaler, SCALER_PATH)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model():
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    print("Initializing model...")
    input_size = X_train[0].shape[0]
    model = VariableLengthSpeakerVerificationModel(input_size)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Starting training...")
    num_epochs = 100
    batch_size = 16
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).unsqueeze(1)
            batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        
        epoch_loss = total_loss / len(X_train)
        epoch_accuracy = correct_predictions / total_predictions
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    print("Training completed. Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return model


def verify_speaker(audio_path, model, scaler):
    """Verify if the audio is from the authorized user."""
    embedding = get_embedding(audio_path)
    embedding_scaled = scaler.transform([embedding])
    
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(embedding_scaled, dtype=torch.float32).unsqueeze(0))
        probability = output.item()
    return probability > 0.8, probability

# def main():
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
#         print("Training new model...")
#         model = train_model()
#         scaler = torch.load(SCALER_PATH)
#     else:
#         print("Loading existing model...")
#         input_size = len(get_embedding(os.path.join(USER_FOLDER, os.listdir(USER_FOLDER)[0])))
#         model = VariableLengthSpeakerVerificationModel(input_size)
#         model.load_state_dict(torch.load(MODEL_PATH))
#         scaler = torch.load(SCALER_PATH)
    
#     # Test the model
#     test_audio_path = "test_audio.wav"
#     is_verified, probability = verify_speaker(test_audio_path, model, scaler)
#     if is_verified:
#         print(f"Speaker verified as the authorized user with probability: {probability:.4f}")
#     else:
#         print(f"Speaker not verified. Probability of being the authorized user: {probability:.4f}")

# if __name__ == "__main__":
#     main()