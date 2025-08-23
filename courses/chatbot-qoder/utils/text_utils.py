"""
Text Utilities Module for Chatbot-Qoder Tutorial Series

This module provides comprehensive text processing utilities including cleaning,
tokenization, vocabulary management, and encoding functions used throughout
the tutorial series.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, Set
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
import json


class SimpleTokenizer:
    """
    Simple tokenizer class for educational purposes.
    
    This tokenizer implements basic word-level tokenization with vocabulary
    management, special token handling, and encoding/decoding capabilities.
    """
    
    def __init__(self, vocab_size: int = 10000, special_tokens: Optional[Dict[str, str]] = None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size (int): Maximum vocabulary size
            special_tokens (Dict[str, str], optional): Special tokens mapping
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>", 
            "sos_token": "<SOS>",
            "eos_token": "<EOS>"
        }
        
        # Initialize vocabulary dictionaries
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_built = False
        
        # Add special tokens first
        for token_name, token_str in self.special_tokens.items():
            self._add_token(token_str)
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts (List[str]): List of text strings
            min_freq (int): Minimum frequency for a token to be included
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count token frequencies
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        print(f"Found {len(token_counts)} unique tokens")
        
        # Sort by frequency and select top tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add tokens to vocabulary (special tokens already added)
        current_vocab_size = len(self.token_to_id)
        for token, freq in sorted_tokens:
            if freq >= min_freq and current_vocab_size < self.vocab_size:
                if token not in self.token_to_id:
                    self._add_token(token)
                    current_vocab_size += 1
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.
        
        Args:
            text (str): Input text to tokenize
        
        Returns:
            List[str]: List of tokens
        """
        # Basic word tokenization
        # Clean and normalize text
        text = text.lower().strip()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text (str): Input text to encode
            add_special_tokens (bool): Whether to add SOS/EOS tokens
            max_length (int, optional): Maximum sequence length
        
        Returns:
            List[int]: List of token IDs
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.special_tokens["sos_token"]] + tokens + [self.special_tokens["eos_token"]]
        
        # Convert to IDs
        unk_id = self.token_to_id[self.special_tokens["unk_token"]]
        token_ids = [self.token_to_id.get(token, unk_id) for token in tokens]
        
        # Truncate or pad if max_length specified
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length:
                pad_id = self.token_to_id[self.special_tokens["pad_token"]]
                token_ids.extend([pad_id] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs
            skip_special_tokens (bool): Whether to skip special tokens
        
        Returns:
            str: Decoded text
        """
        tokens = []
        special_token_strs = set(self.special_tokens.values())
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in special_token_strs:
                    continue
                tokens.append(token)
        
        return " ".join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size."""
        return len(self.token_to_id)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file."""
        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "vocab_built": self.vocab_built
        }
        
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data["vocab_size"],
            special_tokens=tokenizer_data["special_tokens"]
        )
        
        tokenizer.token_to_id = tokenizer_data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer_data["id_to_token"].items()}
        tokenizer.vocab_built = tokenizer_data["vocab_built"]
        
        return tokenizer


class CharacterTokenizer(SimpleTokenizer):
    """Character-level tokenizer for language modeling tasks."""
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters.
        
        Args:
            text (str): Input text to tokenize
        
        Returns:
            List[str]: List of characters
        """
        # Return list of characters (including spaces)
        return list(text)


def clean_text(text: str, 
               lowercase: bool = True,
               remove_punctuation: bool = False,
               remove_special_chars: bool = False,
               remove_extra_whitespace: bool = True,
               normalize_unicode: bool = True) -> str:
    """
    Clean and normalize text according to specified options.
    
    Args:
        text (str): Input text to clean
        lowercase (bool): Convert to lowercase
        remove_punctuation (bool): Remove punctuation marks
        remove_special_chars (bool): Remove special characters
        remove_extra_whitespace (bool): Remove extra whitespace
        normalize_unicode (bool): Normalize unicode characters
    
    Returns:
        str: Cleaned text
    
    Example:
        >>> text = "Hello!!!   How are you? 😊"
        >>> clean_text(text, remove_punctuation=True)
        'hello how are you 😊'
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove special characters (keep only alphanumeric and spaces)
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def build_vocabulary(texts: List[str], 
                    vocab_size: int = 10000,
                    min_freq: int = 2,
                    tokenizer_type: str = "word",
                    special_tokens: Optional[Dict[str, str]] = None) -> SimpleTokenizer:
    """
    Build vocabulary from a collection of texts.
    
    Args:
        texts (List[str]): List of texts to build vocabulary from
        vocab_size (int): Maximum vocabulary size
        min_freq (int): Minimum frequency for inclusion
        tokenizer_type (str): Type of tokenizer ("word" or "char")
        special_tokens (Dict[str, str], optional): Special tokens mapping
    
    Returns:
        SimpleTokenizer: Fitted tokenizer with built vocabulary
    
    Example:
        >>> texts = ["hello world", "hello there", "world peace"]
        >>> tokenizer = build_vocabulary(texts, vocab_size=100)
        >>> print(tokenizer.get_vocab_size())
    """
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Initialize tokenizer
    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    else:
        tokenizer = SimpleTokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    # Build vocabulary
    tokenizer.build_vocabulary(cleaned_texts, min_freq=min_freq)
    
    return tokenizer


def encode_sequences(texts: List[str], 
                    tokenizer: SimpleTokenizer,
                    max_length: Optional[int] = None,
                    add_special_tokens: bool = True) -> List[List[int]]:
    """
    Encode multiple text sequences to token ID sequences.
    
    Args:
        texts (List[str]): List of texts to encode
        tokenizer (SimpleTokenizer): Fitted tokenizer
        max_length (int, optional): Maximum sequence length
        add_special_tokens (bool): Whether to add special tokens
    
    Returns:
        List[List[int]]: List of encoded sequences
    
    Example:
        >>> texts = ["hello world", "goodbye world"]
        >>> tokenizer = build_vocabulary(texts)
        >>> encoded = encode_sequences(texts, tokenizer, max_length=10)
    """
    return [tokenizer.encode(text, add_special_tokens, max_length) for text in texts]


def pad_sequences(sequences: List[List[int]], 
                 max_length: Optional[int] = None,
                 pad_value: int = 0,
                 padding: str = "post",
                 truncating: str = "post") -> torch.Tensor:
    """
    Pad sequences to the same length.
    
    Args:
        sequences (List[List[int]]): List of sequences to pad
        max_length (int, optional): Maximum length (if None, use longest sequence)
        pad_value (int): Value to use for padding
        padding (str): "pre" or "post" padding
        truncating (str): "pre" or "post" truncating
    
    Returns:
        torch.Tensor: Padded sequences tensor of shape (batch_size, max_length)
    
    Example:
        >>> sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> padded = pad_sequences(sequences, max_length=5)
        >>> print(padded.shape)
        torch.Size([3, 5])
    """
    if not sequences:
        return torch.tensor([])
    
    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Initialize padded tensor
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_value, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        # Truncate if necessary
        if seq_len > max_length:
            if truncating == "post":
                seq = seq[:max_length]
            else:  # pre
                seq = seq[-max_length:]
            seq_len = max_length
        
        # Pad sequence
        if padding == "post":
            padded[i, :seq_len] = torch.tensor(seq)
        else:  # pre
            padded[i, -seq_len:] = torch.tensor(seq)
    
    return padded


def create_attention_mask(sequences: torch.Tensor, 
                         pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask for padded sequences.
    
    Args:
        sequences (torch.Tensor): Padded sequences tensor
        pad_token_id (int): ID of the padding token
    
    Returns:
        torch.Tensor: Attention mask (1 for real tokens, 0 for padding)
    
    Example:
        >>> sequences = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        >>> mask = create_attention_mask(sequences, pad_token_id=0)
        >>> print(mask)
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0]])
    """
    return (sequences != pad_token_id).long()


def text_to_sequences(texts: List[str],
                     tokenizer: Optional[SimpleTokenizer] = None,
                     vocab_size: int = 10000,
                     max_length: Optional[int] = None,
                     return_tensors: bool = True) -> Union[List[List[int]], torch.Tensor]:
    """
    Complete pipeline to convert texts to padded sequences.
    
    Args:
        texts (List[str]): List of input texts
        tokenizer (SimpleTokenizer, optional): Pre-fitted tokenizer
        vocab_size (int): Vocabulary size if building new tokenizer
        max_length (int, optional): Maximum sequence length
        return_tensors (bool): Whether to return torch tensors
    
    Returns:
        Union[List[List[int]], torch.Tensor]: Encoded sequences
    
    Example:
        >>> texts = ["hello world", "goodbye world", "hello there"]
        >>> sequences = text_to_sequences(texts, max_length=10)
        >>> print(sequences.shape)
        torch.Size([3, 10])
    """
    # Build or use existing tokenizer
    if tokenizer is None:
        tokenizer = build_vocabulary(texts, vocab_size=vocab_size)
    
    # Encode sequences
    encoded = encode_sequences(texts, tokenizer, max_length=max_length)
    
    if return_tensors:
        # Pad sequences
        pad_id = tokenizer.token_to_id[tokenizer.special_tokens["pad_token"]]
        return pad_sequences(encoded, max_length=max_length, pad_value=pad_id)
    
    return encoded


def calculate_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate statistics about a collection of texts.
    
    Args:
        texts (List[str]): List of texts to analyze
    
    Returns:
        Dict[str, float]: Dictionary containing various text statistics
    
    Example:
        >>> texts = ["hello world", "goodbye cruel world", "hello there"]
        >>> stats = calculate_text_statistics(texts)
        >>> print(stats["avg_length"])
    """
    if not texts:
        return {}
    
    # Calculate basic statistics
    lengths = [len(text.split()) for text in texts]
    char_lengths = [len(text) for text in texts]
    
    # Vocabulary statistics
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    
    unique_tokens = set(all_tokens)
    
    return {
        "num_texts": len(texts),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_char_length": sum(char_lengths) / len(char_lengths),
        "total_tokens": len(all_tokens),
        "unique_tokens": len(unique_tokens),
        "vocabulary_diversity": len(unique_tokens) / len(all_tokens) if all_tokens else 0
    }


# Export commonly used functions and classes
__all__ = [
    "SimpleTokenizer",
    "CharacterTokenizer",
    "clean_text",
    "build_vocabulary",
    "encode_sequences",
    "pad_sequences", 
    "create_attention_mask",
    "text_to_sequences",
    "calculate_text_statistics"
]