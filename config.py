# config.py
"""Configuration for Speech Emotion Recognition"""

# Data paths
RAVDESS_CSV = 'Data/CSV_Data/Ravdess.csv'
TESS_CSV = 'Data/CSV_Data/Tess.csv'
# CREMA_CSV = 'Data/CSV_Data/Crema.csv'
# SAVEE_CSV = 'Data/CSV_Data/Savee.csv'

# Feature paths
FEATURES_PATH = 'Data/Json_features/X_datanew.json'
LABELS_PATH = 'Data/Json_features/Y_datanew.json'
TEST_FEATURES_PATH = 'Data/Json_features/x_test_data.json'
TEST_LABELS_PATH = 'Data/Json_features/y_test_data.json'

# Model paths
MODEL_JSON_PATH = 'Model/model8723.json'
MODEL_WEIGHTS_PATH = 'best_weights.weights.h5'

# Audio parameters
TOTAL_LENGTH = 173056
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

# Model parameters
BATCH_SIZE = 23
EPOCHS = 340
TEST_SPLIT = 0.125
VAL_SPLIT = 0.304

# Emotion mapping
EMOTION_MAP = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fear': 5,
    'disgust': 6,
    'surprise': 7
}

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']