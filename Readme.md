# Speech Emotion Recognition - Modular Code

This is a clean modular version of your notebook code. No extras, just your exact functionality organized into modules.

## Files Structure
```
├── config.py           # All configuration parameters
├── data_preparation.py # Load and prepare data
├── feature_extraction.py # Extract audio features
├── model.py           # LSTM model definition
├── training.py        # Training logic
├── evaluation.py      # Testing and visualization
├── main.py           # Main script
└── requirements.txt   # Dependencies
├── realtime_SER.py    # Real-time speech emotion recognition
```

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Full training pipeline (extract features + train)
```bash
python main.py --mode train
```

### 3. Extract features only (saves for later)
```bash
python main.py --mode extract
```

### 4. Train using saved features (faster)
```bash
python main.py --mode train-saved
```

### 5. Test saved model
```bash
python main.py --mode test
```

### 6. Run realtime_SER
```bash
python realtime_SER.py
```

## What Each Module Does

- **config.py**: All paths and parameters in one place
- **data_preparation.py**: Loads CSVs, merges data, maps emotions
- **feature_extraction.py**: Extracts RMS, ZCR, MFCC features
- **model.py**: LSTM model (64-64-8 architecture)
- **training.py**: Splits data, trains model
- **evaluation.py**: Tests model, plots confusion matrix
- **main.py**: Ties everything together
- **realtime_SER.py**: Run the model to predict in real-time

## Notes

- This is your exact notebook code, just organized into modules
- Features are saved to `Data/Json_features/`
- Model saved to `Model/`
- No extra features or complications added