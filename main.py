# main.py
"""Main script for Speech Emotion Recognition"""

import os
import argparse

def train_pipeline():
    """Complete training pipeline"""
    from data_preparation import prepare_data
    from feature_extraction import extract_features, save_features
    from training import train_model
    from evaluation import evaluate_model, plot_confusion_matrix, plot_history
    
    # Step 1: Load data
    print("Loading data...")
    data = prepare_data()
    
    # Step 2: Extract features
    print("Extracting features...")
    X, Y = extract_features(data)
    save_features(X, Y)
    
    # Step 3: Train model
    print("Training model...")
    model, history = train_model(X, Y)
    
    # Step 4: Evaluate
    print("Evaluating model...")
    y_true, y_pred, acc = evaluate_model(model)
    
    # Step 5: Visualize
    plot_confusion_matrix(y_true, y_pred)
    plot_history(history)
    
    print(f"\nTraining complete! Test accuracy: {acc:.4f}")

def test_pipeline():
    """Test saved model"""
    from model import load_model
    from evaluation import evaluate_model, plot_confusion_matrix
    
    print("Loading saved model...")
    model = load_model()
    
    print("Evaluating model...")
    y_true, y_pred, acc = evaluate_model(model)
    plot_confusion_matrix(y_true, y_pred)
    
    print(f"Test accuracy: {acc:.4f}")

def extract_only():
    """Extract and save features only"""
    from data_preparation import prepare_data
    from feature_extraction import extract_features, save_features
    
    print("Loading data...")
    data = prepare_data()
    
    print("Extracting features...")
    X, Y = extract_features(data)
    save_features(X, Y)
    print("Features saved!")

def train_from_saved():
    """Train using saved features"""
    from feature_extraction import load_features
    from training import train_model
    from evaluation import evaluate_model, plot_confusion_matrix, plot_history
    
    print("Loading saved features...")
    X, Y = load_features()
    
    print("Training model...")
    model, history = train_model(X, Y)
    
    print("Evaluating model...")
    y_true, y_pred, acc = evaluate_model(model)
    
    plot_confusion_matrix(y_true, y_pred)
    plot_history(history)
    
    print(f"Training complete! Test accuracy: {acc:.4f}")

def ensure_directories():
    """Create necessary directories"""
    dirs = ['Data/Json_features', 'Model', 'logs', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition")
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'extract', 'train-saved'],
                       help='Mode to run')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    if args.mode == 'train':
        train_pipeline()
    elif args.mode == 'test':
        test_pipeline()
    elif args.mode == 'extract':
        extract_only()
    elif args.mode == 'train-saved':
        train_from_saved()
    
    print("Done!")