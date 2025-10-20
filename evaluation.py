# evaluation.py
"""Model evaluation and visualization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from json_tricks import load
import config

def evaluate_model(model):
    """Evaluate model on test data"""
    # Load test data
    x_test = load(config.TEST_FEATURES_PATH)
    x_test = np.asarray(x_test).astype('float32')
    
    y_test = load(config.TEST_LABELS_PATH)
    y_test = np.asarray(y_test).astype('int8')
    y_test_class = tf.keras.utils.to_categorical(y_test, 8)
    
    # Evaluate
    loss, acc = model.evaluate(x_test, y_test_class, verbose=2)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    # Get predictions
    predictions = model.predict(x_test)
    y_pred_class = np.argmax(predictions, axis=1)
    y_test_labels = np.argmax(y_test_class, axis=1)
    
    return y_test_labels, y_pred_class, acc

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 8))
    cm_df = pd.DataFrame(cm, config.EMOTION_LABELS, config.EMOTION_LABELS)
    
    ax = plt.axes()
    sns.heatmap(cm_df, ax=ax, cmap='BuGn', fmt="d", annot=True)
    ax.set_ylabel('True emotion')
    ax.set_xlabel('Predicted emotion')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print accuracy per emotion
    values = cm.diagonal()
    row_sum = np.sum(cm, axis=1)
    acc = values / row_sum
    
    print('\nPredicted emotions accuracy:')
    for e in range(len(values)):
        print(f"{config.EMOTION_LABELS[e]}: {acc[e]:0.4f}")

def plot_history(history):
    """Plot training history"""
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss (training)')
    plt.plot(history.history['val_loss'], label='Loss (validation)')
    plt.title('Loss for train and validation')
    plt.ylabel('Loss value')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Acc (training)')
    plt.plot(history.history['val_categorical_accuracy'], label='Acc (validation)')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()