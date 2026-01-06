"""
Transformer-Based Sentiment Classifier

This module implements a complete transformer encoder for sentiment analysis with:
- Multi-head self-attention with attention weight extraction
- Sinusoidal positional encoding
- Pre-layer normalization for stable training
- Support for both custom training and pretrained model fine-tuning
"""

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Positional Encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with attention weight extraction.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head attention.
        
        Args:
            query, key, value: Input tensors (batch_size, seq_len, d_model)
            mask: Attention mask
            return_attention: Whether to return attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        self.attention_weights = attention_weights.detach()
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# =============================================================================
# Transformer Encoder Layer
# =============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-layer normalization.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_output, attention_weights = self.self_attention(
            normed, normed, normed, mask, return_attention
        )
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward with residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x, attention_weights


# =============================================================================
# Main Transformer Classifier
# =============================================================================

class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier for sentiment analysis.
    
    Args:
        vocab_size: Size of vocabulary
        max_seq_length: Maximum sequence length
        embedding_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        feedforward_dim: Feed-forward network dimension
        num_classes: Number of output classes
        dropout: Dropout probability
        pad_token_id: Padding token ID for masking
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_length: int = 256,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # Store attention weights from all layers
        self.attention_weights: Dict[int, torch.Tensor] = {}
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention."""
        mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask.float()
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing logits and optionally attention weights
        """
        if attention_mask is None:
            mask = self.create_padding_mask(input_ids)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
        
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        
        self.attention_weights = {}
        
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x, attn_weights = layer(x, mask, return_attention)
            if return_attention and attn_weights is not None:
                self.attention_weights[i] = attn_weights
        
        x = self.final_norm(x)
        
        # Mean pooling over sequence (excluding padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
        else:
            mask_expanded = (input_ids != self.pad_token_id).unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        logits = self.classifier(x)
        
        output = {"logits": logits}
        if return_attention:
            output["attention_weights"] = self.attention_weights
        
        return output
    
    def get_attention_weights(self, layer: int = -1) -> Optional[torch.Tensor]:
        """Get attention weights from a specific layer."""
        if layer == -1:
            layer = self.num_layers - 1
        return self.attention_weights.get(layer)


# =============================================================================
# Pretrained Model Wrapper (for fine-tuning)
# =============================================================================

class PretrainedTransformerClassifier(nn.Module):
    """
    Fine-tunable pretrained transformer classifier using HuggingFace models.
    
    Args:
        pretrained_model: Name of pretrained model (e.g., 'bert-base-uncased')
        num_classes: Number of output classes
        dropout: Dropout probability
        freeze_encoder: Whether to freeze encoder weights
    """
    
    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.attention_weights: Dict[int, torch.Tensor] = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """Forward pass through pretrained encoder and classifier."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        if return_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            self.attention_weights = {i: attn for i, attn in enumerate(outputs.attentions)}
            result["attention_weights"] = self.attention_weights
        
        return result
    
    def get_attention_weights(self, layer: int = -1) -> Optional[torch.Tensor]:
        """Get attention weights from a specific layer."""
        num_layers = len(self.attention_weights)
        if layer == -1:
            layer = num_layers - 1
        return self.attention_weights.get(layer)


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

import random
from torch.utils.data import Dataset, DataLoader


class TextAugmenter:
    """Text augmentation with synonym replacement, deletion, and swapping."""
    
    SYNONYMS = {
        "good": ["great", "excellent", "fine", "nice", "wonderful"],
        "bad": ["terrible", "awful", "poor", "horrible", "dreadful"],
        "happy": ["joyful", "pleased", "delighted", "cheerful", "glad"],
        "sad": ["unhappy", "sorrowful", "dejected", "gloomy", "melancholy"],
        "love": ["adore", "cherish", "appreciate", "enjoy", "like"],
        "hate": ["despise", "detest", "loathe", "dislike", "abhor"],
        "beautiful": ["gorgeous", "lovely", "stunning", "attractive", "pretty"],
        "amazing": ["incredible", "wonderful", "fantastic", "marvelous", "superb"],
        "boring": ["dull", "tedious", "monotonous", "uninteresting", "tiresome"],
        "funny": ["hilarious", "amusing", "comical", "humorous", "entertaining"],
        "scary": ["frightening", "terrifying", "creepy", "spooky", "horrifying"],
    }
    
    def __init__(self, synonym_prob: float = 0.1, deletion_prob: float = 0.1, swap_prob: float = 0.1):
        self.synonym_prob = synonym_prob
        self.deletion_prob = deletion_prob
        self.swap_prob = swap_prob
    
    def augment(self, text: str) -> str:
        words = text.split()
        
        # Synonym replacement
        for i, word in enumerate(words):
            if random.random() < self.synonym_prob:
                word_lower = word.lower()
                if word_lower in self.SYNONYMS:
                    synonym = random.choice(self.SYNONYMS[word_lower])
                    words[i] = synonym.capitalize() if word[0].isupper() else synonym
        
        # Random deletion
        if len(words) > 1:
            words = [w for w in words if random.random() > self.deletion_prob] or [random.choice(words)]
        
        # Random swap
        for i in range(len(words) - 1):
            if random.random() < self.swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        
        return " ".join(words)


class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        augmenter: Optional[TextAugmenter] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.augmenter is not None:
            text = self.augmenter.augment(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text
        }


def load_dataset(dataset_name: str = "imdb", max_samples: Optional[int] = None):
    """Load IMDb or SST-2 dataset."""
    from datasets import load_dataset as hf_load_dataset
    
    if dataset_name.lower() == "imdb":
        dataset = hf_load_dataset("imdb")
        texts = dataset["train"]["text"] + dataset["test"]["text"]
        labels = dataset["train"]["label"] + dataset["test"]["label"]
    elif dataset_name.lower() == "sst2":
        dataset = hf_load_dataset("glue", "sst2")
        texts = dataset["train"]["sentence"] + dataset["validation"]["sentence"]
        labels = dataset["train"]["label"] + dataset["validation"]["label"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if max_samples is not None:
        indices = random.sample(range(len(texts)), min(max_samples, len(texts)))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return texts, labels


def create_data_splits(texts, labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train, validation, and test sets."""
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    return (
        ([texts[i] for i in train_idx], [labels[i] for i in train_idx]),
        ([texts[i] for i in val_idx], [labels[i] for i in val_idx]),
        ([texts[i] for i in test_idx], [labels[i] for i in test_idx])
    )


def create_dataloaders(config: dict, seed: int = 42):
    """Create train, validation, and test dataloaders."""
    from transformers import AutoTokenizer
    
    texts, labels = load_dataset(
        config["data"]["dataset"],
        config["data"].get("max_samples")
    )
    
    train_data, val_data, test_data = create_data_splits(
        texts, labels,
        config["data"]["train_split"],
        config["data"]["val_split"],
        seed
    )
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    augmenter = None
    if config["augmentation"]["enabled"]:
        augmenter = TextAugmenter(
            config["augmentation"]["synonym_replacement_prob"],
            config["augmentation"]["random_deletion_prob"],
            config["augmentation"]["random_swap_prob"]
        )
    
    train_dataset = SentimentDataset(train_data[0], train_data[1], tokenizer, config["model"]["max_seq_length"], augmenter)
    val_dataset = SentimentDataset(val_data[0], val_data[1], tokenizer, config["model"]["max_seq_length"])
    test_dataset = SentimentDataset(test_data[0], test_data[1], tokenizer, config["model"]["max_seq_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"], pin_memory=True)
    
    return train_loader, val_loader, test_loader, tokenizer


# =============================================================================
# Edge Cases for Evaluation
# =============================================================================

EDGE_CASES = [
    {"text": "I don't think this movie is bad.", "expected": "positive", "category": "negation"},
    {"text": "This is not a good film at all.", "expected": "negative", "category": "negation"},
    {"text": "I wouldn't say I didn't enjoy it.", "expected": "positive", "category": "double_negation"},
    {"text": "Oh great, another superhero movie. Just what we needed.", "expected": "negative", "category": "sarcasm"},
    {"text": "Wow, I totally loved waiting 2 hours for nothing to happen.", "expected": "negative", "category": "sarcasm"},
    {"text": "The acting was brilliant but the plot was terrible.", "expected": "mixed", "category": "mixed_sentiment"},
    {"text": "Despite the poor script, the visuals were stunning.", "expected": "mixed", "category": "mixed_sentiment"},
    {"text": "I walked out after 30 minutes.", "expected": "negative", "category": "implicit"},
    {"text": "I've watched this movie five times already.", "expected": "positive", "category": "implicit"},
    {"text": "Not as bad as I expected.", "expected": "neutral_positive", "category": "comparative"},
    {"text": "Although the first half was slow, the second half completely redeemed the movie.", "expected": "positive", "category": "long_range"},
    {"text": "It was... a movie.", "expected": "negative", "category": "subtle"},
]


def get_edge_cases() -> List[Dict[str, str]]:
    """Get edge case examples for model evaluation."""
    return EDGE_CASES


# =============================================================================
# Factory Function
# =============================================================================

def create_model(config: dict, use_pretrained: bool = False) -> nn.Module:
    """Create model based on configuration."""
    if use_pretrained:
        return PretrainedTransformerClassifier(
            pretrained_model=config.get("pretrained_model", "bert-base-uncased"),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1),
            freeze_encoder=config.get("freeze_encoder", False)
        )
    else:
        return TransformerClassifier(
            vocab_size=config.get("vocab_size", 30522),
            max_seq_length=config.get("max_seq_length", 256),
            embedding_dim=config.get("embedding_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            feedforward_dim=config.get("feedforward_dim", 1024),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.1)
        )
