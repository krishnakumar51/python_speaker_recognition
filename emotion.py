import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained LSTM model
# lstm_model = load_model('lstm_model.h5')  # Adjust the path as needed

def preprocess_audio_for_lstm(file_path, max_length=200):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)
    return mfccs.transpose(1, 0)  # Shape: (timesteps, features)

def predict_with_lstm(file_path):
    audio_data = preprocess_audio_for_lstm(file_path)
    lstm_model = load_model('emotion_model.h5') 
    predictions = lstm_model.predict(np.expand_dims(audio_data, axis=0))
    
    # Check the shape of predictions
    print("Predictions shape:", predictions.shape)

    # Emotion labels corresponding to the classes in your dataset
    emotion_labels = ['fear','normal']
    num_classes = len(emotion_labels)  # Should be 5 in this case

    # Get predicted class
    predicted_class = np.argmax(predictions, axis=1)
    
    # Convert class index to label
    predicted_emotion_labels = [emotion_labels[i] for i in predicted_class]

    return predicted_class, predicted_emotion_labels

# Example usage
if __name__ == "__main__":
    input_file = "pw.wav"  # Adjust the path as needed
    predicted_class, predicted_emotion = predict_with_lstm(input_file)
    print("Predicted class (LSTM):", predicted_class)
    print("Predicted emotion (LSTM):", predicted_emotion)
