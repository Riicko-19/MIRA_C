"""
Utility functions for MIRA Wave Person C diagnostic system.
"""

import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two feature vectors."""
    return np.linalg.norm(vec1 - vec2)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calculate cosine similarity between two vectors with numerical safety.
    
    Args:
        vec1: First vector
        vec2: Second vector
        eps: Minimum norm threshold to avoid division by zero
        
    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is near-zero
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    norm_product = norm1 * norm2
    
    if norm_product < eps:
        return 0.0
    
    similarity = dot_product / norm_product
    # Ensure result is in valid range due to floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def normalize_probabilities(probs: Dict[str, float], eps: float = 1e-10) -> Dict[str, float]:
    """
    Normalize a probability distribution to sum to 1.0 with robust handling.
    
    Args:
        probs: Dictionary of probabilities (may contain NaN, inf, or negative values)
        eps: Minimum total to avoid division by zero
        
    Returns:
        Normalized probability distribution
    """
    if not probs:
        return probs
    
    # Convert non-finite values to 0.0 and clip negatives
    clean_probs = {}
    for k, v in probs.items():
        if np.isfinite(v) and v > 0:
            clean_probs[k] = float(v)
        else:
            clean_probs[k] = 0.0
    
    total = sum(clean_probs.values())
    
    # If total is too small, return uniform distribution
    if total < eps:
        n = len(clean_probs)
        return {k: 1.0 / n for k in clean_probs.keys()}
    
    # Normalize
    return {k: v / total for k, v in clean_probs.items()}


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file."""
    # Convert numpy types to native Python types
    serializable_data = convert_to_serializable(data)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ==============================================================================
# NUMERICAL ROBUSTNESS HELPERS (Phase 1)
# ==============================================================================

def safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute standard deviation with protection against zero variance.
    
    Args:
        x: Input array
        eps: Minimum std to return (default 1e-8)
        
    Returns:
        Standard deviation, or eps if std < eps
    """
    std = np.std(x)
    return std if std >= eps else eps


def safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize vector (z-score) with protection against zero variance.
    
    Args:
        vec: Input vector
        eps: Minimum std for normalization
        
    Returns:
        Normalized vector
    """
    mean = np.mean(vec)
    std = safe_std(vec, eps)
    return (vec - mean) / std


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Pearson correlation coefficient with NaN/inf protection.
    
    This helper wraps np.corrcoef to:
    - Silence RuntimeWarnings for zero-variance features
    - Replace NaN/inf with 0.0 (no correlation)
    
    Args:
        x: First array
        y: Second array
        eps: Not used (for API consistency)
        
    Returns:
        Correlation coefficient in [-1, 1], or 0.0 if undefined
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(x, y)
        corr_value = corr_matrix[0, 1]
        
        # Replace non-finite values with 0.0
        if not np.isfinite(corr_value):
            return 0.0
        
        return float(corr_value)


def safe_division(numerator: float, denominator: float, default: float = 0.0, eps: float = 1e-10) -> float:
    """
    Safe division with protection against division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator ~ 0
        eps: Minimum denominator threshold
        
    Returns:
        numerator / denominator, or default if denominator too small
    """
    if abs(denominator) < eps:
        return default
    return numerator / denominator


def ensure_finite(value: float, default: float = 0.0) -> float:
    """
    Ensure a value is finite (not NaN or inf).
    
    Args:
        value: Input value
        default: Default value if non-finite
        
    Returns:
        Original value if finite, otherwise default
    """
    return value if np.isfinite(value) else default
