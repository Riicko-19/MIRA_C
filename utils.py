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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0
    return dot_product / norm_product


def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize a probability distribution to sum to 1.0."""
    total = sum(probs.values())
    if total == 0:
        return probs
    return {k: v / total for k, v in probs.items()}


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
