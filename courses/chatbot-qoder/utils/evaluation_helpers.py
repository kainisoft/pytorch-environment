"""
Evaluation Helpers Module for Chatbot-Qoder Tutorial Series

This module provides utilities for evaluating chatbot models including
BLEU score, perplexity, response relevance, and conversation quality metrics.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from collections import Counter
import re
import math


def bleu_score(reference: List[str], 
               candidate: List[str], 
               n_grams: int = 4,
               weights: Optional[List[float]] = None) -> float:
    """
    Calculate BLEU score for a candidate against reference.
    
    Args:
        reference (List[str]): Reference tokens
        candidate (List[str]): Candidate tokens
        n_grams (int): Maximum n-gram order
        weights (List[float], optional): Weights for different n-gram orders
    
    Returns:
        float: BLEU score
    
    Example:
        >>> ref = ["the", "cat", "is", "on", "the", "mat"]
        >>> cand = ["the", "cat", "sits", "on", "the", "mat"]
        >>> score = bleu_score(ref, cand)
    """
    if weights is None:
        weights = [1.0 / n_grams] * n_grams
    
    if len(candidate) == 0 or len(reference) == 0:
        return 0.0
    
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, n_grams + 1):
        ref_ngrams = Counter()
        cand_ngrams = Counter()
        
        # Generate n-grams
        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i + n])
            ref_ngrams[ngram] += 1
        
        for i in range(len(candidate) - n + 1):
            ngram = tuple(candidate[i:i + n])
            cand_ngrams[ngram] += 1
        
        # Calculate precision
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
        else:
            matches = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) 
                         for ngram in cand_ngrams)
            precision = matches / sum(cand_ngrams.values())
            precisions.append(precision)
    
    # Calculate weighted geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bleu = brevity_penalty * math.exp(log_precision)
    
    return bleu


def rouge_score(reference: List[str], 
                candidate: List[str], 
                rouge_type: str = "rouge-1") -> Dict[str, float]:
    """
    Calculate ROUGE score for a candidate against reference.
    
    Args:
        reference (List[str]): Reference tokens
        candidate (List[str]): Candidate tokens
        rouge_type (str): Type of ROUGE ("rouge-1", "rouge-2", "rouge-l")
    
    Returns:
        Dict[str, float]: ROUGE scores (precision, recall, f1)
    
    Example:
        >>> ref = ["the", "cat", "is", "on", "the", "mat"]
        >>> cand = ["the", "cat", "sits", "on", "the", "mat"]
        >>> scores = rouge_score(ref, cand, "rouge-1")
    """
    if len(reference) == 0 or len(candidate) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if rouge_type == "rouge-1":
        ref_ngrams = Counter(reference)
        cand_ngrams = Counter(candidate)
    elif rouge_type == "rouge-2":
        ref_ngrams = Counter(tuple(reference[i:i+2]) for i in range(len(reference)-1))
        cand_ngrams = Counter(tuple(candidate[i:i+2]) for i in range(len(candidate)-1))
    elif rouge_type == "rouge-l":
        # Longest Common Subsequence
        lcs_length = _lcs_length(reference, candidate)
        precision = lcs_length / len(candidate) if len(candidate) > 0 else 0.0
        recall = lcs_length / len(reference) if len(reference) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    else:
        raise ValueError(f"Unknown ROUGE type: {rouge_type}")
    
    # Calculate precision and recall
    overlap = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) for ngram in cand_ngrams)
    precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0.0
    recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def perplexity(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> float:
    """
    Calculate perplexity of a language model.
    
    Args:
        model (torch.nn.Module): Language model
        dataloader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to run on
    
    Returns:
        float: Perplexity score
    
    Example:
        >>> ppl = perplexity(model, test_loader, device)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = inputs[:, 1:]  # Shift for next token prediction
                inputs = inputs[:, :-1]
            
            outputs = model(inputs)
            
            # Flatten for loss calculation
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(outputs_flat, targets_flat, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += targets_flat.numel()
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def response_relevance(query: str, 
                      response: str,
                      method: str = "jaccard") -> float:
    """
    Calculate relevance of response to query.
    
    Args:
        query (str): Input query
        response (str): Generated response
        method (str): Similarity method ("jaccard", "cosine")
    
    Returns:
        float: Relevance score between 0 and 1
    
    Example:
        >>> query = "What is the weather like?"
        >>> response = "The weather is sunny today."
        >>> score = response_relevance(query, response)
    """
    # Tokenize and clean
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    response_tokens = set(re.findall(r'\b\w+\b', response.lower()))
    
    if len(query_tokens) == 0 or len(response_tokens) == 0:
        return 0.0
    
    if method == "jaccard":
        intersection = len(query_tokens.intersection(response_tokens))
        union = len(query_tokens.union(response_tokens))
        return intersection / union if union > 0 else 0.0
    
    elif method == "cosine":
        # Simple cosine similarity based on word overlap
        all_tokens = query_tokens.union(response_tokens)
        query_vector = [1 if token in query_tokens else 0 for token in all_tokens]
        response_vector = [1 if token in response_tokens else 0 for token in all_tokens]
        
        dot_product = sum(q * r for q, r in zip(query_vector, response_vector))
        query_norm = math.sqrt(sum(q * q for q in query_vector))
        response_norm = math.sqrt(sum(r * r for r in response_vector))
        
        return dot_product / (query_norm * response_norm) if query_norm * response_norm > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def conversation_quality(conversation: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Evaluate quality of a conversation.
    
    Args:
        conversation (List[Tuple[str, str]]): List of (query, response) pairs
    
    Returns:
        Dict[str, float]: Quality metrics
    
    Example:
        >>> conv = [("Hello", "Hi there!"), ("How are you?", "I'm doing well, thanks!")]
        >>> quality = conversation_quality(conv)
    """
    if not conversation:
        return {"coherence": 0.0, "engagement": 0.0, "diversity": 0.0}
    
    # Calculate coherence (average relevance between adjacent turns)
    coherence_scores = []
    for i in range(len(conversation) - 1):
        _, prev_response = conversation[i]
        query, _ = conversation[i + 1]
        coherence_scores.append(response_relevance(prev_response, query))
    
    coherence = np.mean(coherence_scores) if coherence_scores else 0.0
    
    # Calculate engagement (response length and question asking)
    response_lengths = [len(response.split()) for _, response in conversation]
    avg_response_length = np.mean(response_lengths)
    
    # Normalize length score (optimal around 10-20 words)
    length_score = min(avg_response_length / 15.0, 1.0)
    
    # Count questions asked by the system
    questions_asked = sum(1 for _, response in conversation if '?' in response)
    question_ratio = questions_asked / len(conversation)
    
    engagement = (length_score + question_ratio) / 2
    
    # Calculate diversity (unique words in responses)
    all_response_words = []
    for _, response in conversation:
        all_response_words.extend(re.findall(r'\b\w+\b', response.lower()))
    
    if all_response_words:
        unique_words = len(set(all_response_words))
        total_words = len(all_response_words)
        diversity = unique_words / total_words
    else:
        diversity = 0.0
    
    return {
        "coherence": coherence,
        "engagement": engagement,
        "diversity": diversity,
        "overall": (coherence + engagement + diversity) / 3
    }


def calculate_metrics(predictions: List[str],
                     references: List[str],
                     tokenize: bool = True) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        predictions (List[str]): Predicted texts
        references (List[str]): Reference texts
        tokenize (bool): Whether to tokenize texts
    
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    
    Example:
        >>> preds = ["the cat sits on the mat"]
        >>> refs = ["the cat is on the mat"]
        >>> metrics = calculate_metrics(preds, refs)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        if tokenize:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
        else:
            pred_tokens = pred
            ref_tokens = ref
        
        # BLEU score
        bleu = bleu_score(ref_tokens, pred_tokens)
        bleu_scores.append(bleu)
        
        # ROUGE scores
        rouge_1 = rouge_score(ref_tokens, pred_tokens, "rouge-1")
        rouge_2 = rouge_score(ref_tokens, pred_tokens, "rouge-2")
        rouge_l = rouge_score(ref_tokens, pred_tokens, "rouge-l")
        
        rouge_1_scores.append(rouge_1["f1"])
        rouge_2_scores.append(rouge_2["f1"])
        rouge_l_scores.append(rouge_l["f1"])
    
    return {
        "bleu": np.mean(bleu_scores),
        "rouge_1": np.mean(rouge_1_scores),
        "rouge_2": np.mean(rouge_2_scores),
        "rouge_l": np.mean(rouge_l_scores)
    }


def human_evaluation_interface(conversations: List[List[Tuple[str, str]]],
                              criteria: List[str] = None) -> Dict[str, List[float]]:
    """
    Simple interface for human evaluation of conversations.
    
    Args:
        conversations (List[List[Tuple[str, str]]]): List of conversations
        criteria (List[str], optional): Evaluation criteria
    
    Returns:
        Dict[str, List[float]]: Human evaluation scores
    
    Note:
        This is a simplified interface for educational purposes.
        In practice, you would use more sophisticated evaluation platforms.
    """
    if criteria is None:
        criteria = ["relevance", "fluency", "informativeness", "coherence"]
    
    print("Human Evaluation Interface")
    print("=" * 40)
    print("Rate each conversation on a scale of 1-5 for each criterion")
    print()
    
    scores = {criterion: [] for criterion in criteria}
    
    for i, conversation in enumerate(conversations):
        print(f"\nConversation {i + 1}:")
        print("-" * 20)
        
        for turn_idx, (query, response) in enumerate(conversation):
            print(f"User: {query}")
            print(f"Bot: {response}")
            if turn_idx < len(conversation) - 1:
                print()
        
        print("\nPlease rate this conversation:")
        conv_scores = {}
        
        for criterion in criteria:
            while True:
                try:
                    score = input(f"{criterion.capitalize()} (1-5): ").strip()
                    score = float(score)
                    if 1 <= score <= 5:
                        conv_scores[criterion] = score
                        break
                    else:
                        print("Please enter a score between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        # Add scores to results
        for criterion, score in conv_scores.items():
            scores[criterion].append(score)
        
        print()
    
    # Calculate average scores
    avg_scores = {criterion: np.mean(score_list) 
                 for criterion, score_list in scores.items()}
    
    print("\nAverage Scores:")
    for criterion, avg_score in avg_scores.items():
        print(f"{criterion.capitalize()}: {avg_score:.2f}")
    
    return scores


# Export commonly used functions
__all__ = [
    "bleu_score",
    "rouge_score",
    "perplexity",
    "response_relevance",
    "conversation_quality",
    "calculate_metrics",
    "human_evaluation_interface"
]