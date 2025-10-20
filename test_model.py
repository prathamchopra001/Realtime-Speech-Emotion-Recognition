# test_model.py
"""Quick test of the trained model"""

import numpy as np
from json_tricks import load
import tensorflow as tf
import config

def test_saved_model():
    """Test the saved model performance"""
    print("Loading saved model...")
    
    # Load model
    from model import load_model
    model = load_model()
    
    # Load test data
    print("Loading test data...")
    x_test = load(config.TEST_FEATURES_PATH)
    x_test = np.asarray(x_test).astype('float32')
    
    y_test = load(config.TEST_LABELS_PATH)
    y_test = np.asarray(y_test).astype('int8')
    y_test_class = tf.keras.utils.to_categorical(y_test, 8)
    
    # Evaluate
    print("Evaluating...")
    loss, acc = model.evaluate(x_test, y_test_class, verbose=0)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Get predictions for confusion matrix
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test_class, axis=1)
    
    # Calculate per-class accuracy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nPer-Class Accuracy:")
    print("-"*30)
    for i, emotion in enumerate(config.EMOTION_LABELS):
        if cm[i].sum() > 0:
            accuracy = cm[i, i] / cm[i].sum()
            print(f"{emotion:12s}: {accuracy:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    print("="*50)
    
    # Show some predictions
    print("\nSample Predictions (first 5):")
    for i in range(min(5, len(y_true))):
        true_emotion = config.EMOTION_LABELS[y_true[i]]
        pred_emotion = config.EMOTION_LABELS[y_pred[i]]
        confidence = predictions[i][y_pred[i]]
        correct = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"{correct} True: {true_emotion:10s} | Predicted: {pred_emotion:10s} (conf: {confidence:.2%})")

if __name__ == "__main__":
    test_saved_model()