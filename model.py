# model.py
"""LSTM model for emotion recognition"""

from keras.models import Sequential, model_from_json
from keras import layers, callbacks
import tensorflow as tf
import config

def build_model(input_shape):
    """Build the LSTM model"""
    model = Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=(input_shape[1:3])))
    model.add(layers.Dropout(0.3)) 
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.3)) 
    model.add(layers.Dense(8, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='RMSProp',
        metrics=['categorical_accuracy']
    )
    
    print(model.summary())
    return model

def get_callbacks():
    """Create training callbacks"""
    checkpoint = callbacks.ModelCheckpoint(
        config.MODEL_WEIGHTS_PATH,
        save_best_only=True,
        monitor='val_categorical_accuracy',
        mode='max'
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.1,
        patience=20
    )
    
    return [checkpoint, reduce_lr]

def save_model(model):
    """Save model architecture and weights"""
    model_json = model.to_json()
    with open(config.MODEL_JSON_PATH, "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(config.MODEL_WEIGHTS_PATH)
    print("Model saved")

def load_model():
    """Load saved model"""
    with open(config.MODEL_JSON_PATH, 'r') as json_file:
        json_savedModel = json_file.read()
    
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(config.MODEL_WEIGHTS_PATH)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='RMSProp',
        metrics=['categorical_accuracy']
    )
    
    return model