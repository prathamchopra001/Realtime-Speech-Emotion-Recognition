# data_preparation.py
"""Data loading and preprocessing"""

import pandas as pd
import numpy as np
import config

def load_data():
    """Load and merge RAVDESS and TESS datasets"""
    ravdess = pd.read_csv(config.RAVDESS_CSV)
    tess = pd.read_csv(config.TESS_CSV)
    # crema = pd.read_csv(config.CREMA_CSV)
    # savee = pd.read_csv(config.SAVEE_CSV)
    data = pd.concat([ravdess, tess], ignore_index=True)
    data.dropna(inplace=True)
    return data

def map_emotions(data):
    """Map emotion strings to integers"""
    data['Emotions'] = data['Emotions'].apply(lambda e: config.EMOTION_MAP.get(e, -1))
    return data

def prepare_data():
    """Complete data preparation pipeline"""
    data = load_data()
    data = map_emotions(data)
    print(f"Loaded {len(data)} samples")
    print(f"Unique emotions: {data['Emotions'].unique()}")
    return data