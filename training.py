# training.py
"""Model training"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import os
import config
from model import build_model, get_callbacks, save_model

def prepare_splits(X, y):
    """Split data into train, validation, and test sets"""
    x_train, x_tosplit, y_train, y_tosplit = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=1
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tosplit, y_tosplit, test_size=config.VAL_SPLIT, random_state=1
    )
    
    # Convert to categorical
    y_train_class = tf.keras.utils.to_categorical(y_train, 8)
    y_val_class = tf.keras.utils.to_categorical(y_val, 8)
    
    print(f"Train shape: {x_train.shape}")
    print(f"Val shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")
    
    # Save test data
    os.makedirs('Data/Json_features', exist_ok=True)
    with open(config.TEST_FEATURES_PATH, 'w') as f:
        json.dump(x_test.tolist(), f)
    with open(config.TEST_LABELS_PATH, 'w') as f:
        json.dump(y_test.tolist(), f)
    
    return x_train, x_val, x_test, y_train_class, y_val_class, y_test

def train_model(X, y):
    """Train the model"""
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_splits(X, y)
    
    # Build model
    model = build_model(X.shape)
    
    # Get callbacks
    callback_list = get_callbacks()
    
    # Train
    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=callback_list
    )
    
    # Try to load best weights if they exist
    if os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"\nAttempting to load best weights from {config.MODEL_WEIGHTS_PATH}")
        try:
            # Build a fresh model with same architecture
            fresh_model = build_model(X.shape)
            # Load the weights
            fresh_model.load_weights(config.MODEL_WEIGHTS_PATH)
            model = fresh_model
            print("Successfully loaded best weights!")
        except Exception as e:
            print(f"Warning: Could not load best weights: {e}")
            print("Using final epoch weights instead.")
    else:
        print("No checkpoint found, using final epoch weights.")
    
    # Save final model
    print("\nSaving final model...")
    save_model(model)
    
    return model, history