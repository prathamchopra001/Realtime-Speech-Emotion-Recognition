import numpy as np
import pyaudio
import librosa
import noisereduce as nr
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import config
from model import load_model

class RealtimeEmotionRecognizer:
    def __init__(self):
        """Initialize real-time emotion recognizer"""
        print("Loading model...")
        self.model = load_model()
        print("Model loaded successfully!")
        
        # Audio parameters
        self.RATE = 22050  # Sample rate
        self.CHUNK = 1024  # Chunks to read at once
        self.RECORD_SECONDS = 3  # Duration of audio to analyze
        self.BUFFER_SIZE = int(self.RATE * self.RECORD_SECONDS)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # Thread-safe queue for predictions
        self.prediction_queue = queue.Queue()
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Control flags
        self.recording = False
        self.analyzing = False
        
    def extract_features_from_buffer(self, audio_data):
        """Extract features from audio buffer"""
        try:
            # Convert to numpy array
            audio = np.array(audio_data, dtype='float32')
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Trim silence
            xt, _ = librosa.effects.trim(audio, top_db=30)
            
            # Pad or truncate to fixed length
            if len(xt) > config.TOTAL_LENGTH:
                xt = xt[:config.TOTAL_LENGTH]
            else:
                xt = np.pad(xt, (0, config.TOTAL_LENGTH - len(xt)), 'constant')
            
            # Simple noise reduction
            xt = nr.reduce_noise(y=xt, sr=self.RATE)
            
            # Extract features
            rms = librosa.feature.rms(y=xt, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
            zcr = librosa.feature.zero_crossing_rate(xt, frame_length=config.FRAME_LENGTH, 
                                                     hop_length=config.HOP_LENGTH, center=True)
            mfcc = librosa.feature.mfcc(y=xt, sr=self.RATE, n_mfcc=config.N_MFCC, hop_length=config.HOP_LENGTH)
            
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
            print(f"Feature extraction error: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def analyze_emotion(self):
        """Analyze emotion from audio buffer"""
        while self.recording:
            if len(self.audio_buffer) >= self.BUFFER_SIZE:
                # Get audio data
                audio_data = list(self.audio_buffer)
                
                # Extract features
                features = self.extract_features_from_buffer(audio_data)
                
                if features is not None:
                    # Predict emotion
                    predictions = self.model.predict(features, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    confidence = predictions[0][emotion_idx]
                    
                    # Get top 3 emotions
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    top_3 = [(config.EMOTION_LABELS[i], predictions[0][i]) for i in top_3_idx]
                    
                    result = {
                        'emotion': config.EMOTION_LABELS[emotion_idx],
                        'confidence': confidence,
                        'top_3': top_3,
                        'timestamp': time.time()
                    }
                    
                    self.prediction_queue.put(result)
            
            time.sleep(0.5)  # Analyze every 0.5 seconds
    
    def start(self):
        """Start real-time recognition"""
        print("\n" + "="*50)
        print("REAL-TIME EMOTION RECOGNITION")
        print("="*50)
        print(f"Listening for {self.RECORD_SECONDS} second segments...")
        print("Press Ctrl+C to stop\n")
        
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        self.recording = True
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.analyze_emotion)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Start stream
        self.stream.start_stream()
        
        # Display results
        try:
            last_emotion = None
            while True:
                if not self.prediction_queue.empty():
                    result = self.prediction_queue.get()
                    
                    emotion = result['emotion']
                    confidence = result['confidence']
                    
                    # Display with emotion indicator
                    emotion_display = self.get_emotion_display(emotion)
                    
                    if emotion != last_emotion or confidence > 0.7:
                        print(f"\r{emotion_display} {emotion.upper():10s} ({confidence:.1%}) ", end="")
                        
                        # Show top 3 if confidence is low
                        if confidence < 0.5:
                            print("\n  Top 3: ", end="")
                            for e, c in result['top_3']:
                                print(f"{e}:{c:.1%} ", end="")
                            print()
                    
                    last_emotion = emotion
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.stop()
    
    def get_emotion_display(self, emotion):
        """Get emoji for emotion"""
        emoji_map = {
            'neutral': '😐',
            'calm': '😌',
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fearful': '😨',
            'fear': '😨',
            'disgust': '🤢',
            'surprised': '😲',
            'surprise': '😲'
        }
        return emoji_map.get(emotion, '🎭')
    
    def stop(self):
        """Stop real-time recognition"""
        print("\n\nStopping...")
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.p.terminate()
        print("Stopped successfully!")


def main():
    recognizer = RealtimeEmotionRecognizer()
    recognizer.start()


if __name__ == "__main__":
    main()