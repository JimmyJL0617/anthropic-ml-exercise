# Anthropic ML Technical Exercise
This repository contains a complete suite of solutions for the Anthropic Machine Learning technical assessment. The project spans custom Transformer implementation, research analysis on AI safety, code optimization, and empirical scaling law modeling.

·Repository Structure

anthropic-ml-exercise/
├── README.md                   # Project overview and execution guide
├── requirements.txt            # Dependency specifications
├── part1_implementation/       # Custom Transformer & Sentiment Analysis
│   ├── model.py                # Transformer Encoder architecture
│   ├── train.py                # Training loop with WandB integration
│   ├── evaluate.py             # Attention visualization and metrics
│   ├── config.yaml             # Hyperparameter management
│   └── report.md               # Technical findings and ablation study
├── part2_research_analysis/    # AI Safety & Reliability
│   └── technical_document.md   # Analysis of hallucinations, bias, and consistency
├── part3_code_review/          # Optimization & Best Practices
│   ├── review.md               # Detailed critique of the provided snippet
│   └── improved_code.py        # Refactored, production-ready implementation
└── bonus_scaling_laws/         # Empirical Modeling
    └── scaling_analysis.ipynb  # Jupyter notebook with Chinchilla-style fitting

·Setup Instructions

1. Environment Preparation
It is recommended to use a virtual environment (Python 3.9+):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Logging Setup (Weights & Biases)
For Part 1, metrics are logged via WandB. Log in to your account:
wandb login

·Execution Guide

Part 1: Model Implementation & Training
This section trains a custom Transformer Encoder on the IMDb dataset.
To Train:
python -m part1_implementation.train
To Evaluate & Visualize Attention:
python -m part1_implementation.evaluate

Part 2: Research Analysis
The technical document addressing AI Assistant reliability (Hallucination, Bias, etc.) is located at part2_research_analysis/technical_document.md. It covers:
-Root cause analysis of prompt sensitivity.
-Experimental design for RLHF (Reinforcement Learning from Human Feedback) to reduce contradictions.

Part 3: Code Review & Optimization
This folder contains the critique of the baseline Transformer implementation.
Review Findings: Documentation of critical bugs (e.g., missing zero_grad(), lack of Positional Encodings).
Improved Code: Run the refactored script to verify the fix:
python part3_code_review/improved_code.py

Bonus: Scaling Law Analysis
The Jupyter Notebook provides a power-law fit for model performance.
-To View: Open bonus_scaling_laws/scaling_analysis.ipynb in Jupyter Lab/Notebook.
-Key Prediction: 10B model trained on 1T tokens and optimal compute allocation for 20 PF-days.

·Design Philosophy
-Modular Engineering: Each part is self-contained with decoupled logic to ensure maintainability.
-Safety & Interpretability: Emphasis is placed on why the model makes decisions (Attention Maps) and how to mitigate risks (Part 2).
-Rigor: Implementation follows industry standards, including gradient clipping, AdamW optimization, and type hinting.
