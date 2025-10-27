# Fake Review Detection

A PyTorch-based fake review detection system using RoBERTa (Robustly Optimized BERT Pretraining Approach) with comprehensive feature engineering and ablation studies.

## Overview

This project provides a complete machine learning pipeline for detecting fake reviews using state-of-the-art transformer models (RoBERTa) combined with linguistic, behavioral, and metadata features. The system supports feature ablation studies to understand the contribution of different feature types.

## Features

- **RoBERTa-based Classification**: Leverages pretrained RoBERTa models for text understanding
- **Multi-Feature Fusion**: Combines text embeddings with:
  - Linguistic features (sentiment, lexical diversity, writing style)
  - Behavioral features (rating patterns, reviewer history)
  - Metadata features (temporal patterns, review length)
- **Feature Ablation Studies**: Analyze the contribution of different feature groups
- **Hydra Configuration**: Flexible configuration management with Hydra
- **Experiment Tracking**: Support for Weights & Biases and TensorBoard
- **Comprehensive Testing**: Unit tests for all major components

## Project Structure

```
fake-review-detection/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/                  # Data configurations
│   ├── model/                 # Model configurations
│   └── training/              # Training configurations
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   └── processed/             # Preprocessed data
├── experiments/               # Experiment outputs
│   ├── checkpoints/          # Model checkpoints
│   └── logs/                 # Training logs
├── scripts/                   # Executable scripts
│   ├── preprocess_data.py    # Data preprocessing
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   └── feature_ablation.py   # Feature ablation study
├── src/fake_review_detection/ # Source code
│   ├── data/                 # Data processing modules
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   └── preprocessing.py  # Text preprocessing
│   ├── models/               # Model architectures
│   │   ├── roberta_classifier.py  # RoBERTa classifier
│   │   └── trainer.py        # Training utilities
│   ├── evaluation/           # Evaluation utilities
│   │   └── metrics.py        # Metrics calculation
│   └── features/             # Feature extraction
│       ├── linguistic.py     # Linguistic features
│       ├── behavioral.py     # Behavioral features
│       ├── metadata.py       # Metadata features
│       └── ablation.py       # Ablation study utilities
├── tests/                     # Unit tests
├── Makefile                   # Common commands
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chantaldelcarmen/fake-review-detection.git
cd fake-review-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
make install
# Or manually:
pip install -r requirements.txt
```

4. Install development dependencies (optional):
```bash
make install-dev
```

## Usage

### 1. Data Preparation

Prepare your data in CSV format with `text` and `label` columns:
- `text`: Review text
- `label`: 0 for genuine, 1 for fake

Place your raw data in `data/raw/reviews.csv`.

### 2. Data Preprocessing

```bash
make preprocess
# Or:
python scripts/preprocess_data.py
```

This will:
- Clean and preprocess text
- Split data into train/validation/test sets
- Save processed data to `data/processed/`

### 3. Model Training

```bash
make train
# Or:
python scripts/train.py
```

Configure training parameters in `configs/training/default.yaml` or via command-line:
```bash
python scripts/train.py training.num_epochs=10 training.learning_rate=1e-5
```

### 4. Model Evaluation

```bash
make evaluate
# Or:
python scripts/evaluate.py
```

### 5. Feature Ablation Study

```bash
make feature-ablation
# Or:
python scripts/feature_ablation.py
```

## Configuration

The project uses Hydra for configuration management. Main configuration files:

- `configs/config.yaml`: Main configuration
- `configs/data/default.yaml`: Data processing settings
- `configs/model/roberta_base.yaml`: Model architecture settings
- `configs/training/default.yaml`: Training hyperparameters

### Example: Changing Model

```bash
python scripts/train.py model=roberta_large
```

### Example: Custom Configuration

```bash
python scripts/train.py \
    training.num_epochs=10 \
    training.learning_rate=2e-5 \
    data.batch_size=32 \
    model.dropout_rate=0.2
```

## Development

### Running Tests

```bash
make test
# Or:
pytest tests/ -v
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

### Cleaning Build Artifacts

```bash
make clean
```

## Model Architecture

The system uses a RoBERTa-based architecture with optional feature fusion:

1. **Text Encoder**: RoBERTa (base or large) for text representation
2. **Feature Fusion**: Concatenates text embeddings with engineered features
3. **Classification Head**: Fully connected layer for binary classification

## Feature Engineering

### Linguistic Features
- Word and sentence counts
- Average word/sentence length
- Sentiment polarity and subjectivity
- Lexical diversity
- Punctuation patterns
- Capitalization patterns

### Behavioral Features
- Rating patterns
- Helpful vote counts
- Verified purchase status
- Reviewer history statistics

### Metadata Features
- Temporal patterns (day of week, hour)
- Review length
- Product categories

## Experiment Tracking

### Weights & Biases

Enable W&B tracking in `configs/config.yaml`:
```yaml
use_wandb: true
wandb_project: fake-review-detection
wandb_entity: your-entity
```

### TensorBoard

Logs are automatically saved to `experiments/logs/`. View with:
```bash
tensorboard --logdir experiments/logs
```

## Performance Tips

1. **GPU Usage**: Set `device: cuda` in config for GPU acceleration
2. **Batch Size**: Increase batch size for faster training (if memory allows)
3. **Mixed Precision**: Enable `use_fp16: true` for faster training on modern GPUs
4. **Gradient Accumulation**: Use `gradient_accumulation_steps` for larger effective batch sizes

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting
6. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fake_review_detection,
  author = {Your Name},
  title = {Fake Review Detection using RoBERTa},
  year = {2024},
  url = {https://github.com/chantaldelcarmen/fake-review-detection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- RoBERTa model by Liu et al.
- PyTorch framework

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.