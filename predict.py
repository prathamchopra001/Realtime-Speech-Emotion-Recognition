# predict.py
"""Predict emotion from a single audio file"""

import sys
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
import config
from model import load_model

def extract_single_audio_features(file_path):
    """Extract features from a single audio file"""
    try:
        # Load audio
        _, sr = librosa.load(path=file_path, sr=None)
        rawsound = AudioSegment.from_file(file_path)
        
        # Normalize
        normalizedsound = effects.normalize(rawsound, headroom=0)
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
        
        # Trim silence
        xt, _ = librosa.effects.trim(normal_x, top_db=30)
        
        # Pad or truncate
        if len(xt) > config.TOTAL_LENGTH:
            xt = xt[:config.TOTAL_LENGTH]
        padded_x = np.pad(xt, (0, config.TOTAL_LENGTH - len(xt)), 'constant')
        
        # Noise reduction
        final_x = nr.reduce_noise(y=padded_x, sr=sr)
        
        # Extract features
        rms = librosa.feature.rms(y=final_x, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=final_x, frame_length=config.FRAME_LENGTH, 
                                                hop_length=config.HOP_LENGTH, center=True)
        mfcc = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=config.N_MFCC, hop_length=config.HOP_LENGTH)
        
        # Reshape
        f_rms = np.asarray([rms]).astype('float32')
        f_rms = np.swapaxes(f_rms, 1, 2)
        f_zcr = np.asarray([zcr]).astype('float32')
        f_zcr = np.swapaxes(f_zcr, 1, 2)
        f_mfccs = np.asarray([mfcc]).astype('float32')
        f_mfccs = np.swapaxes(f_mfccs, 1, 2)
        
        # Concatenate
        X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)
        return X
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    # Extract features
    print(f"Processing: {audio_path}")
    features = extract_single_audio_features(audio_path)
    
    if features is None:
        print("Failed to extract features")
        return
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Predict
    predictions = model.predict(features, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    
    # Results
    print("\n=== Prediction Results ===")
    print(f"Predicted emotion: {config.EMOTION_LABELS[emotion_idx]}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nAll probabilities:")
    for i, label in enumerate(config.EMOTION_LABELS):
        print(f"  {label}: {predictions[0][i]:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    predict_emotion(audio_file)