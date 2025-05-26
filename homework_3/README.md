# Advanced Text Retrieval System

This project implements various text retrieval approaches for question-answering tasks, featuring both traditional and neural-based methods.

## Project Structure

- `models/` - Contains different retrieval model implementations
  - `tfidf_retriever.py` - TF-IDF based retrieval
  - `neural_retriever.py` - Neural network based retrieval using transformers
- `utils/` - Utility functions
  - `data_processor.py` - Data loading and preprocessing
  - `metrics.py` - Evaluation metrics
- `config.py` - Configuration settings
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `main.py` - Main entry point

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
# For training
python train.py --model_type neural --epochs 3

# For evaluation
python evaluate.py --model_type neural
```

## Features

- Multiple retrieval approaches (TF-IDF and Neural)
- Support for different embedding models
- Comprehensive evaluation metrics
- Easy to extend with new retrieval methods 