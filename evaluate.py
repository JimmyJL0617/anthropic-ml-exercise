#!/usr/bin/env python3
"""
Evaluation Script for Transformer Sentiment Classifier

This script provides comprehensive model evaluation including:
- Multiple metrics (accuracy, F1, precision, recall, AUC-ROC)
- Confusion matrix analysis
- Error analysis with examples
- Edge case evaluation
- Attention visualization

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --visualize-attention
    python evaluate.py --checkpoint checkpoints/best.pt --output-dir results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model, create_dataloaders, get_edge_cases


# =============================================================================
# Utility Functions
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(preferred: str = "cuda") -> torch.device:
    """Get the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Evaluator Class
# =============================================================================

class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str] = ["negative", "positive"]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, return_predictions: bool = False) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_texts = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]
            texts = batch.get("text", [""] * len(labels))
            
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs["logits"], dim=-1)
            predictions = outputs["logits"].argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_texts.extend(texts)
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "precision": precision_score(all_labels, all_predictions, average="weighted", zero_division=0),
            "recall": recall_score(all_labels, all_predictions, average="weighted", zero_division=0),
            "f1": f1_score(all_labels, all_predictions, average="weighted", zero_division=0),
            "f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        }
        
        # AUC-ROC for binary classification
        if len(self.class_names) == 2:
            try:
                metrics["auc_roc"] = roc_auc_score(all_labels, all_probabilities[:, 1])
            except ValueError:
                metrics["auc_roc"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        metrics["classification_report"] = classification_report(
            all_labels, all_predictions, target_names=self.class_names, output_dict=True
        )
        
        if return_predictions:
            metrics["predictions"] = {
                "predicted": all_predictions.tolist(),
                "probabilities": all_probabilities.tolist(),
                "labels": all_labels.tolist(),
                "texts": all_texts
            }
        
        return metrics
    
    def analyze_errors(self, dataloader: DataLoader, num_examples: int = 10) -> Dict[str, List[Dict]]:
        """Analyze model errors with example texts."""
        false_positives = []
        false_negatives = []
        confident_errors = []
        uncertain_correct = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing errors"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                texts = batch.get("text", [""] * len(labels))
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs["logits"], dim=-1)
                predictions = outputs["logits"].argmax(dim=-1).cpu()
                confidence = probabilities.max(dim=-1).values.cpu()
                
                for i in range(len(labels)):
                    pred = predictions[i].item()
                    label = labels[i].item()
                    conf = confidence[i].item()
                    text = texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i]
                    
                    example = {
                        "text": text,
                        "predicted": self.class_names[pred],
                        "actual": self.class_names[label],
                        "confidence": round(conf, 4)
                    }
                    
                    if pred != label:
                        if pred == 1 and label == 0 and len(false_positives) < num_examples:
                            false_positives.append(example)
                        elif pred == 0 and label == 1 and len(false_negatives) < num_examples:
                            false_negatives.append(example)
                        if conf > 0.9 and len(confident_errors) < num_examples:
                            confident_errors.append(example)
                    else:
                        if conf < 0.6 and len(uncertain_correct) < num_examples:
                            uncertain_correct.append(example)
        
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "confident_errors": confident_errors,
            "uncertain_correct": uncertain_correct
        }
    
    def evaluate_edge_cases(self, edge_cases: List[Dict], tokenizer) -> Dict[str, Any]:
        """Evaluate model on edge cases."""
        results = []
        category_results = defaultdict(list)
        
        with torch.no_grad():
            for case in edge_cases:
                text = case["text"]
                expected = case["expected"]
                category = case.get("category", "unknown")
                
                encoding = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs["logits"], dim=-1)
                prediction = outputs["logits"].argmax(dim=-1).item()
                predicted_label = self.class_names[prediction]
                
                # Check correctness (handle mixed/neutral cases)
                correct = predicted_label.lower() == expected.lower() or expected in ["mixed", "neutral_positive", "neutral_negative"]
                
                result = {
                    "text": text,
                    "expected": expected,
                    "predicted": predicted_label,
                    "confidence": probabilities[0, prediction].item(),
                    "category": category,
                    "correct": correct
                }
                
                results.append(result)
                category_results[category].append(result)
        
        # Category statistics
        category_stats = {}
        for cat, cat_results in category_results.items():
            correct = sum(1 for r in cat_results if r["correct"])
            total = len(cat_results)
            category_stats[cat] = {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0}
        
        return {
            "results": results,
            "category_stats": category_stats,
            "overall_accuracy": sum(1 for r in results if r["correct"]) / len(results)
        }


# =============================================================================
# Attention Visualization
# =============================================================================

class AttentionVisualizer:
    """Visualize attention weights from transformer models."""
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def get_attention_weights(self, text: str, layer: int = -1) -> Tuple[torch.Tensor, List[str], Dict]:
        """Extract attention weights for a given text."""
        encoding = self.tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_length = attention_mask[0].sum().item()
        tokens = tokens[:seq_length]
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, return_attention=True)
        
        attn_weights = self.model.get_attention_weights(layer)
        if attn_weights is None:
            raise ValueError("Model did not return attention weights")
        
        attn_weights = attn_weights[0, :, :seq_length, :seq_length]
        return attn_weights, tokens, outputs
    
    def plot_attention_heatmap(self, text: str, layer: int = -1, head: Optional[int] = None, 
                                figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None):
        """Plot attention heatmap for a text."""
        attn_weights, tokens, _ = self.get_attention_weights(text, layer)
        
        if head is not None:
            attn_weights = attn_weights[head]
            title = f"Attention Weights (Layer {layer}, Head {head})"
        else:
            attn_weights = attn_weights.mean(dim=0)
            title = f"Average Attention Weights (Layer {layer})"
        
        attn_weights = attn_weights.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens, cmap="Blues", square=True, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Keys (Attended To)", fontsize=12)
        ax.set_ylabel("Queries (Attending From)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved attention heatmap to {save_path}")
        
        return fig
    
    def plot_attention_on_text(self, text: str, layer: int = -1, figsize: Tuple[int, int] = (14, 3),
                                save_path: Optional[str] = None):
        """Visualize attention as highlighted text."""
        attn_weights, tokens, outputs = self.get_attention_weights(text, layer)
        
        # Average over heads and get attention from CLS token
        attn_weights = attn_weights.mean(dim=0).cpu().numpy()
        token_importance = attn_weights[0]  # Attention from first token
        
        # Normalize
        token_importance = (token_importance - token_importance.min()) / (token_importance.max() - token_importance.min() + 1e-8)
        
        # Get prediction
        probs = F.softmax(outputs["logits"], dim=-1)
        pred = probs.argmax(dim=-1).item()
        pred_label = "Positive" if pred == 1 else "Negative"
        confidence = probs[0, pred].item()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        x_pos = 0.02
        y_pos = 0.5
        cmap = plt.cm.Reds
        
        for token, importance in zip(tokens, token_importance):
            if token in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            
            display_token = token.replace("##", "")
            color = cmap(importance)
            
            text_obj = ax.text(x_pos, y_pos, display_token + " ", fontsize=11, fontfamily="monospace",
                               verticalalignment="center", bbox=dict(facecolor=color, edgecolor="none", pad=2, alpha=0.7))
            
            renderer = fig.canvas.get_renderer()
            bbox = text_obj.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transData.inverted())
            x_pos = bbox_data.x1 + 0.005
            
            if x_pos > 0.95:
                x_pos = 0.02
                y_pos -= 0.25
        
        ax.text(0.02, 0.95, f"Prediction: {pred_label} ({confidence:.1%})", fontsize=11, fontweight="bold", transform=ax.transAxes)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved attention visualization to {save_path}")
        
        return fig
    
    def create_attention_report(self, texts: List[str], output_dir: str, layer: int = -1):
        """Create attention visualizations for multiple texts."""
        output_path = ensure_dir(output_dir)
        
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i + 1}/{len(texts)}")
            try:
                self.plot_attention_heatmap(text, layer, save_path=str(output_path / f"heatmap_{i}.png"))
                plt.close()
                self.plot_attention_on_text(text, layer, save_path=str(output_path / f"text_{i}.png"))
                plt.close()
            except Exception as e:
                print(f"Error processing text {i}: {e}")
        
        print(f"\nAttention report saved to {output_dir}")


# =============================================================================
# Reporting Functions
# =============================================================================

def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if "auc_roc" in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  Predicted:  Neg   Pos")
    print(f"  Actual Neg: {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"  Actual Pos: {cm[1, 0]:5d} {cm[1, 1]:5d}")
    print("=" * 50)


def save_evaluation_report(metrics: Dict, error_analysis: Dict, edge_case_results: Dict, output_path: str) -> None:
    """Save comprehensive evaluation report."""
    report = {
        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
        "error_analysis": error_analysis,
        "edge_case_evaluation": edge_case_results
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Evaluation report saved to {output_path}")


# =============================================================================
# Ablation Study
# =============================================================================

ABLATION_CONFIGS = [
    {"name": "baseline", "description": "Baseline configuration", "changes": {}},
    {"name": "heads_4", "description": "4 attention heads", "changes": {"model": {"num_heads": 4}}},
    {"name": "heads_16", "description": "16 attention heads", "changes": {"model": {"num_heads": 16}}},
    {"name": "layers_2", "description": "2 transformer layers", "changes": {"model": {"num_layers": 2}}},
    {"name": "layers_6", "description": "6 transformer layers", "changes": {"model": {"num_layers": 6}}},
    {"name": "no_augmentation", "description": "Without data augmentation", "changes": {"augmentation": {"enabled": False}}},
    {"name": "dropout_0.2", "description": "Higher dropout (0.2)", "changes": {"model": {"dropout": 0.2}}},
    {"name": "dropout_0.05", "description": "Lower dropout (0.05)", "changes": {"model": {"dropout": 0.05}}},
]


def run_ablation_study(base_config: Dict, output_dir: str = "results/ablation", quick: bool = False):
    """Run ablation study to test different configurations."""
    from train import Trainer, TrainingConfig, EarlyStoppingConfig, set_seed
    
    output_path = ensure_dir(output_dir)
    results = []
    device = get_device(base_config.get("device", "cuda"))
    
    def merge_config(base, changes):
        result = base.copy()
        for key, value in changes.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    for ablation in ABLATION_CONFIGS:
        name = ablation["name"]
        print(f"\n{'='*60}\nRunning ablation: {name}\n{'='*60}")
        
        config = merge_config(base_config, ablation["changes"])
        if quick:
            config["training"]["epochs"] = 3
            config["data"]["max_samples"] = 5000
        
        try:
            set_seed(config.get("seed", 42))
            train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config, seed=config.get("seed", 42))
            model = create_model(config["model"])
            
            training_config = TrainingConfig(
                batch_size=config["training"]["batch_size"],
                learning_rate=config["training"]["learning_rate"],
                epochs=config["training"]["epochs"],
                save_dir=f"checkpoints/ablation/{name}",
                log_dir=f"logs/ablation/{name}",
                experiment_name=f"ablation_{name}",
                device=str(device)
            )
            
            trainer = Trainer(model, train_loader, val_loader, training_config)
            history = trainer.train()
            
            evaluator = Evaluator(model, device)
            test_metrics = evaluator.evaluate(test_loader)
            
            results.append({
                "name": name,
                "description": ablation["description"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "parameters": sum(p.numel() for p in model.parameters())
            })
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append({"name": name, "error": str(e)})
    
    # Save results
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}\nABLATION STUDY SUMMARY\n{'='*70}")
    print(f"{'Experiment':<20} {'Test Acc':>10} {'Test F1':>10} {'Parameters':>12}")
    print("-" * 55)
    for r in results:
        if "error" not in r:
            print(f"{r['name']:<20} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} {r['parameters']:>12,}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer Sentiment Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--visualize-attention", action="store_true", help="Generate attention visualizations")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--num-attention-samples", type=int, default=10, help="Number of attention samples")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--ablation-quick", action="store_true", help="Quick ablation with fewer epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))
    print(f"Using device: {device}")
    
    # Run ablation study if requested
    if args.ablation:
        run_ablation_study(config, args.output_dir + "/ablation", quick=args.ablation_quick)
        return
    
    # Create dataloaders
    print("\nLoading data...")
    _, _, test_loader, tokenizer = create_dataloaders(config, seed=config.get("seed", 42))
    
    # Load model
    print("\nLoading model...")
    model = create_model(config["model"])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    evaluator = Evaluator(model, device)
    
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    test_metrics = evaluator.evaluate(test_loader, return_predictions=True)
    print_evaluation_summary(test_metrics)
    
    # Error analysis
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)
    error_analysis = evaluator.analyze_errors(test_loader)
    
    print("\nFalse Positives:")
    for ex in error_analysis["false_positives"][:3]:
        print(f"  [{ex['confidence']:.1%}] {ex['text'][:80]}...")
    
    print("\nFalse Negatives:")
    for ex in error_analysis["false_negatives"][:3]:
        print(f"  [{ex['confidence']:.1%}] {ex['text'][:80]}...")
    
    print("\nConfident Errors (>90% confidence):")
    for ex in error_analysis["confident_errors"][:3]:
        print(f"  [{ex['confidence']:.1%}] Pred: {ex['predicted']}, Actual: {ex['actual']}")
    
    # Edge cases
    print("\n" + "=" * 50)
    print("EDGE CASE EVALUATION")
    print("=" * 50)
    edge_cases = get_edge_cases()
    edge_results = evaluator.evaluate_edge_cases(edge_cases, tokenizer)
    
    print(f"\nOverall: {edge_results['overall_accuracy']:.1%}")
    for cat, stats in edge_results["category_stats"].items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    
    print("\nDetailed results:")
    for r in edge_results["results"]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} [{r['category']}] {r['predicted']} ({r['confidence']:.1%}) | Expected: {r['expected']}")
        print(f"      {r['text'][:60]}...")
    
    # Attention visualization
    if args.visualize_attention:
        print("\n" + "=" * 50)
        print("ATTENTION VISUALIZATION")
        print("=" * 50)
        
        viz_dir = ensure_dir(Path(args.output_dir) / "attention")
        visualizer = AttentionVisualizer(model, tokenizer, device)
        
        sample_texts = []
        for batch in test_loader:
            sample_texts.extend(batch["text"][:args.num_attention_samples])
            if len(sample_texts) >= args.num_attention_samples:
                break
        
        visualizer.create_attention_report(sample_texts[:args.num_attention_samples], str(viz_dir))
    
    # Save report
    ensure_dir(args.output_dir)
    save_evaluation_report(test_metrics, error_analysis, edge_results, f"{args.output_dir}/evaluation_report.json")
    
    print(f"\nReport saved to: {args.output_dir}/evaluation_report.json")


if __name__ == "__main__":
    main()
