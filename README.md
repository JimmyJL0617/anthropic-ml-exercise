# Anthropic ML Engineering Exercise

A comprehensive machine learning exercise demonstrating transformer implementation, research analysis, code review skills, and scaling law analysis.

## Repository Structure

```
anthropic-ml-exercise/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── part1_implementation/              # Transformer sentiment classifier
│   ├── model.py                       # Model architecture & data loading
│   ├── train.py                       # Training loop & infrastructure
│   ├── evaluate.py                    # Evaluation & visualization
│   ├── config.yaml                    # Hyperparameters & settings
│   └── report.md                      # Technical report (1-2 pages)
├── part2_research_analysis/           # Research & problem solving
│   └── technical_document.pdf         # Analysis document (3-4 pages)
├── part3_code_review/                 # Code review exercise
│   ├── review.md                      # Detailed code review
│   └── improved_code.py               # Refactored implementation
└── bonus_scaling_laws/                # Optional scaling analysis
    └── scaling_analysis.ipynb         # Jupyter notebook with analysis
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training, optional)
- 8GB+ RAM (16GB recommended for full dataset)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anthropic-ml-exercise.git
cd anthropic-ml-exercise

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Part 1: Model Implementation & Training

A custom transformer-based sentiment classifier for IMDb movie reviews.

### Features

- Custom transformer encoder with multi-head attention
- Training loop with validation and early stopping
- Attention weight visualization
- Edge case analysis
- Ablation study
- TensorBoard logging
- Checkpoint saving/resuming

### Running Part 1

```bash
cd part1_implementation

# Train the model (full training)
python train.py --config config.yaml

# Train with pretrained BERT (fine-tuning)
python train.py --config config.yaml --use-pretrained

# Debug mode (quick test with limited data)
python train.py --config config.yaml --debug

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/last.pt
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint checkpoints/best.pt

# With attention visualization
python evaluate.py --checkpoint checkpoints/best.pt --visualize-attention

# Run ablation study
python evaluate.py --checkpoint checkpoints/best.pt --ablation

# Quick ablation (fewer epochs)
python evaluate.py --checkpoint checkpoints/best.pt --ablation --ablation-quick
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser at http://localhost:6006
```

### Configuration

All hyperparameters are in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.embedding_dim` | 256 | Token embedding dimension |
| `model.num_heads` | 8 | Attention heads |
| `model.num_layers` | 4 | Transformer layers |
| `training.batch_size` | 32 | Batch size |
| `training.learning_rate` | 1e-4 | Learning rate |
| `training.epochs` | 20 | Max training epochs |
| `early_stopping.patience` | 3 | Early stopping patience |

### Expected Results

| Metric | Custom Transformer | Fine-tuned BERT |
|--------|-------------------|-----------------|
| Accuracy | 85-87% | 92-94% |
| F1 Score | 0.85-0.87 | 0.92-0.94 |
| AUC-ROC | 0.92-0.94 | 0.97-0.98 |

### Output Files

After training:
- `checkpoints/best.pt` - Best model checkpoint
- `checkpoints/last.pt` - Latest checkpoint
- `logs/` - TensorBoard logs
- `results/evaluation_report.json` - Evaluation metrics
- `results/attention/` - Attention visualizations (if enabled)

---

## Part 2: Research Analysis & Problem Solving

Technical document analyzing conversational AI safety issues.

### Document Contents

1. **Problem Analysis** - Root causes of inconsistency, hallucination, bias, and prompt sensitivity
2. **Proposed Solutions** - Technical approaches for top priority issues
3. **Experimental Design** - Detailed experiment plan with statistical analysis
4. **Broader Implications** - Trade-offs and user communication

### Viewing

```bash
# Open the PDF document
open part2_research_analysis/technical_document.pdf  # Mac
# or: xdg-open part2_research_analysis/technical_document.pdf  # Linux
# or: start part2_research_analysis/technical_document.pdf  # Windows
```

---

## Part 3: Code Review & Optimization

Detailed review of a transformer implementation with improvements.

### Contents

- `review.md` - Comprehensive code review covering:
  - Correctness issues (bugs, missing components)
  - Performance optimizations
  - Best practices
  - Documentation needs

- `improved_code.py` - Refactored implementation with:
  - Bug fixes (gradient zeroing, positional encoding)
  - Type hints and documentation
  - Proper device handling
  - Learning rate scheduling
  - Mixed precision training
  - Checkpoint support

### Viewing

```bash
# Read the code review
cat part3_code_review/review.md

# View improved implementation
cat part3_code_review/improved_code.py
```

### Running Improved Code

```bash
cd part3_code_review

# The improved_code.py is self-contained
python improved_code.py
```

---

## Bonus: Scaling Law Analysis

Analysis of neural scaling laws with compute-optimal training predictions.

### Running the Analysis

```bash
cd bonus_scaling_laws

# Start Jupyter notebook
jupyter notebook scaling_analysis.ipynb
```

### Analysis Contents

1. **Scaling Law Fitting** - Fit Chinchilla-style scaling laws to provided data
2. **Loss Prediction** - Predict performance for 10B parameter model on 1T tokens
3. **Compute Allocation** - Optimal model/data split for 20 PF-days budget
4. **Assumptions & Limitations** - Discussion of methodology

---

## Dependencies

### Core Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.14.0
pyyaml>=6.0
tqdm>=4.65.0
```

### Optional (for bonus)

```
jupyter>=1.0.0
scipy>=1.10.0
```

### Install All

```bash
pip install -r requirements.txt
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU ok) | NVIDIA with 8+ GB VRAM |
| Storage | 5 GB | 20+ GB |

### Training Time Estimates

| Configuration | GPU (RTX 3090) | CPU |
|--------------|----------------|-----|
| Debug mode | ~5 min | ~30 min |
| Full training (custom) | ~2 hours | ~12 hours |
| Full training (BERT) | ~4 hours | Not recommended |
| Ablation study (quick) | ~1 hour | ~6 hours |

---

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

**Slow data loading:**
```bash
# Reduce workers or disable pin_memory
data:
  num_workers: 0
  pin_memory: false
```

**Dataset download fails:**
```bash
# Manually download and cache
python -c "from datasets import load_dataset; load_dataset('imdb')"
```

**TensorBoard not showing data:**
```bash
# Ensure logs directory exists and contains events
ls logs/transformer_sentiment/
```

---

## File Descriptions

### Part 1

| File | Lines | Description |
|------|-------|-------------|
| `model.py` | ~500 | Transformer architecture, attention, data loading, augmentation |
| `train.py` | ~400 | Training loop, optimizer, scheduler, checkpointing, logging |
| `evaluate.py` | ~500 | Metrics, error analysis, attention visualization, ablation |
| `config.yaml` | ~80 | All hyperparameters and settings |
| `report.md` | ~150 | Technical report summarizing approach and findings |

### Part 2

| File | Pages | Description |
|------|-------|-------------|
| `technical_document.pdf` | 3-4 | Research analysis of AI safety issues |

### Part 3

| File | Description |
|------|-------------|
| `review.md` | Detailed code review with identified issues |
| `improved_code.py` | Refactored implementation addressing all issues |

### Bonus

| File | Description |
|------|-------------|
| `scaling_analysis.ipynb` | Jupyter notebook with scaling law analysis |

---

## References

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
3. Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
4. Kaplan et al. "Scaling Laws for Neural Language Models" (2020)
5. Maas et al. "Learning Word Vectors for Sentiment Analysis" (IMDb Dataset, 2011)

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact jimmyjl980617@gmail.com.
