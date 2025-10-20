# feature_extraction.py
"""Feature extraction from audio files"""

import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects
from json_tricks import dump, load
import time
import config

def extract_features(data):
    """Extract features from all audio files"""
    rms = []
    zcr = []
    mfcc = []
    emotions = []
    
    print(f"Extracting features from {len(data)} files...")
    tic = time.perf_counter()
    
    for index, row in data.iterrows():
        file_path = row['Path']
        emotion = row['Emotions']
        
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
            f1 = librosa.feature.rms(y=final_x, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
            f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=config.FRAME_LENGTH, 
                                                    hop_length=config.HOP_LENGTH, center=True)
            f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=config.N_MFCC, hop_length=config.HOP_LENGTH)
            
            rms.append(f1)
            zcr.append(f2)
            mfcc.append(f3)
            emotions.append(emotion)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    toc = time.perf_counter()
    print(f"Feature extraction time: {(toc - tic)/60:0.4f} minutes")
    
    # Reshape features
    f_rms = np.asarray(rms).astype('float32')
    f_rms = np.swapaxes(f_rms, 1, 2)
    f_zcr = np.asarray(zcr).astype('float32')
    f_zcr = np.swapaxes(f_zcr, 1, 2)
    f_mfccs = np.asarray(mfcc).astype('float32')
    f_mfccs = np.swapaxes(f_mfccs, 1, 2)
    
    # Concatenate features
    X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)
    Y = np.asarray(emotions).astype('int8')
    Y = np.expand_dims(Y, axis=1)
    
    print(f'Features shape: {X.shape}, Labels shape: {Y.shape}')
    return X, Y

def save_features(X, Y):
    """Save features to JSON files"""
    x_data = X.tolist()
    dump(obj=x_data, fp=config.FEATURES_PATH)
    
    y_data = Y.tolist()
    dump(obj=y_data, fp=config.LABELS_PATH)
    print("Features saved")

def load_features():
    """Load features from JSON files"""
    x = load(config.FEATURES_PATH)
    x = np.asarray(x, dtype='float32')
    
    y = load(config.LABELS_PATH)
    y = np.asarray(y, dtype='int8')
    
    print(f"Features loaded: {x.shape}")
    return x, y