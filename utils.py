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


# ==============================================================================
# CENTRAL INPUT VALIDATION GATE (Phase 2)
# ==============================================================================

def validate_diagnostic_input(run_data: Dict[str, Any]) -> tuple:
    """
    Central authoritative validation for diagnostic input data.
    
    This is the single validation gate that ALL inputs must pass through
    before any agent processing. Validates:
    - Required fields presence
    - Feature value ranges (physics-based limits)
    - Data quality warnings (sensor saturation, anomalies)
    
    Args:
        run_data: Diagnostic run dictionary with structure:
            {
                "run_id": str,
                "fault_location": {"x": float, "y": float},
                "features": {
                    "dominant_frequency": float,
                    "rms_vibration": float,
                    "spectral_entropy": float,
                    "bearing_energy_band": float,
                    "audio_anomaly_score": float,
                    "speed_dependency": str
                },
                "metadata": dict (optional)
            }
    
    Returns:
        Tuple of (is_valid: bool, errors: List[str], warnings: List[str])
        - is_valid: False if any hard errors found
        - errors: List of validation failures (hard stops)
        - warnings: List of suspicious patterns (proceed with caution)
    """
    errors = []
    warnings = []
    
    # ========== Required Fields ==========
    if "run_id" not in run_data:
        errors.append("Missing required field: run_id")
    
    if "fault_location" not in run_data:
        errors.append("Missing required field: fault_location")
    elif not isinstance(run_data["fault_location"], dict):
        errors.append("fault_location must be a dictionary")
    else:
        loc = run_data["fault_location"]
        if "x" not in loc or "y" not in loc:
            errors.append("fault_location must contain 'x' and 'y' coordinates")
        else:
            # Validate coordinate ranges
            if not (0 <= loc["x"] <= 1):
                errors.append(f"fault_location.x={loc['x']} out of range [0, 1]")
            if not (0 <= loc["y"] <= 1):
                errors.append(f"fault_location.y={loc['y']} out of range [0, 1]")
    
    if "features" not in run_data:
        errors.append("Missing required field: features")
        # Cannot continue without features
        return (False, errors, warnings)
    
    features = run_data["features"]
    
    # ========== Feature Validation (Hard Errors) ==========
    
    # 1. Dominant Frequency (1-5000 Hz typical automotive range)
    freq = features.get("dominant_frequency", None)
    if freq is None:
        errors.append("Missing required feature: dominant_frequency")
    elif not isinstance(freq, (int, float)):
        errors.append(f"dominant_frequency must be numeric, got {type(freq).__name__}")
    elif not (1 <= freq <= 5000):
        errors.append(f"dominant_frequency={freq} out of valid range [1, 5000] Hz")
    
    # 2. RMS Vibration (0-50 m/s² physical limit, >15 is extreme)
    rms = features.get("rms_vibration", None)
    if rms is None:
        errors.append("Missing required feature: rms_vibration")
    elif not isinstance(rms, (int, float)):
        errors.append(f"rms_vibration must be numeric, got {type(rms).__name__}")
    elif not (0 <= rms <= 50):
        errors.append(f"rms_vibration={rms} out of valid range [0, 50] m/s²")
    
    # 3. Spectral Entropy (0-2.0 theoretical max, >1.0 unusual for automotive)
    entropy = features.get("spectral_entropy", None)
    if entropy is None:
        errors.append("Missing required feature: spectral_entropy")
    elif not isinstance(entropy, (int, float)):
        errors.append(f"spectral_entropy must be numeric, got {type(entropy).__name__}")
    elif not (0 <= entropy <= 2.0):
        errors.append(f"spectral_entropy={entropy} out of valid range [0, 2.0]")
    
    # 4. Bearing Energy Band (0-1 normalized)
    bearing = features.get("bearing_energy_band", None)
    if bearing is None:
        errors.append("Missing required feature: bearing_energy_band")
    elif not isinstance(bearing, (int, float)):
        errors.append(f"bearing_energy_band must be numeric, got {type(bearing).__name__}")
    elif not (0 <= bearing <= 1):
        errors.append(f"bearing_energy_band={bearing} must be in range [0, 1]")
    
    # 5. Audio Anomaly Score (0-1 normalized)
    audio = features.get("audio_anomaly_score", None)
    if audio is None:
        errors.append("Missing required feature: audio_anomaly_score")
    elif not isinstance(audio, (int, float)):
        errors.append(f"audio_anomaly_score must be numeric, got {type(audio).__name__}")
    elif not (0 <= audio <= 1):
        errors.append(f"audio_anomaly_score={audio} must be in range [0, 1]")
    
    # 6. Speed Dependency (categorical)
    speed_dep = features.get("speed_dependency", None)
    if speed_dep is None:
        errors.append("Missing required feature: speed_dependency")
    elif speed_dep not in ["weak", "medium", "strong"]:
        errors.append(f"speed_dependency='{speed_dep}' must be one of: weak, medium, strong")
    
    # ========== Data Quality Warnings (Soft Issues) ==========
    
    # Only check warnings if no hard errors
    if not errors:
        # Sensor saturation warning
        if rms is not None and rms > 10:
            warnings.append(f"Very high rms_vibration={rms} - check sensor saturation")
        
        # Low entropy warning (stuck sensor or pure tone)
        if entropy is not None and entropy < 0.05:
            warnings.append(f"Very low spectral_entropy={entropy} - possible sensor issue")
        
        # Unusual feature combinations
        if freq is not None and bearing is not None:
            if freq < 500 and bearing > 0.8:
                warnings.append(
                    f"Low frequency ({freq} Hz) with high bearing energy ({bearing}) is unusual"
                )
        
        # Very high entropy (noisy/chaotic signal)
        if entropy is not None and entropy > 0.95:
            warnings.append(f"Very high spectral_entropy={entropy} - signal may be too noisy")
        
        # Extreme audio anomaly
        if audio is not None and audio > 0.95:
            warnings.append(f"Extreme audio_anomaly_score={audio} - verify microphone data")
    
    # Determine overall validity
    is_valid = len(errors) == 0
    
    return (is_valid, errors, warnings)
