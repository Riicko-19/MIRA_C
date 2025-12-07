"""
Input validation for real sensor data in MIRA Wave Person C.

Add quality checks before processing real data through the pipeline.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


def validate_feature_quality(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that real sensor features are within expected ranges.
    
    Args:
        features: Feature dictionary from real sensors
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Dominant frequency check
    freq = features.get("dominant_frequency", 0)
    if freq < 1 or freq > 5000:
        issues.append(f"Dominant frequency {freq} Hz out of range [1, 5000]")
    
    # RMS vibration check
    rms = features.get("rms_vibration", 0)
    if rms < 0 or rms > 10:
        issues.append(f"RMS vibration {rms} m/s² out of range [0, 10]")
    
    # Spectral entropy check
    entropy = features.get("spectral_entropy", 0)
    if entropy < 0 or entropy > 1:
        issues.append(f"Spectral entropy {entropy} out of range [0, 1]")
    
    # Bearing energy band check
    bearing = features.get("bearing_energy_band", 0)
    if bearing < 0 or bearing > 1:
        issues.append(f"Bearing energy band {bearing} out of range [0, 1]")
    
    # Audio anomaly score check
    audio = features.get("audio_anomaly_score", 0)
    if audio < 0 or audio > 1:
        issues.append(f"Audio anomaly score {audio} out of range [0, 1]")
    
    # Speed dependency check
    speed_dep = features.get("speed_dependency", "")
    if speed_dep not in ["weak", "medium", "strong"]:
        issues.append(f"Speed dependency '{speed_dep}' not in ['weak', 'medium', 'strong']")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_fault_location(location: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate fault location coordinates.
    
    Args:
        location: Fault location dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    x = location.get("x", -1)
    y = location.get("y", -1)
    
    if x < 0 or x > 1:
        issues.append(f"Location x={x} out of range [0, 1]")
    
    if y < 0 or y > 1:
        issues.append(f"Location y={y} out of range [0, 1]")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def detect_sensor_anomalies(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect potential sensor issues or data quality problems.
    
    Args:
        features: Feature dictionary
        
    Returns:
        Dictionary with anomaly flags and recommendations
    """
    anomalies = {
        "has_anomalies": False,
        "flags": [],
        "recommendations": []
    }
    
    # Check for suspiciously perfect values (possible sensor saturation)
    if features.get("rms_vibration", 0) >= 9.9:
        anomalies["has_anomalies"] = True
        anomalies["flags"].append("vibration_saturation")
        anomalies["recommendations"].append("Check accelerometer range settings")
    
    # Check for zero variance (sensor stuck)
    if features.get("spectral_entropy", 1) < 0.05:
        anomalies["has_anomalies"] = True
        anomalies["flags"].append("low_signal_variance")
        anomalies["recommendations"].append("Verify sensor connection and sampling rate")
    
    # Check for unrealistic combinations
    freq = features.get("dominant_frequency", 0)
    bearing_energy = features.get("bearing_energy_band", 0)
    
    if freq < 500 and bearing_energy > 0.8:
        anomalies["has_anomalies"] = True
        anomalies["flags"].append("frequency_energy_mismatch")
        anomalies["recommendations"].append(
            "Low frequency with high bearing energy is unusual - verify feature extraction"
        )
    
    return anomalies


def preprocess_real_data(run_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and preprocess real sensor data before pipeline processing.
    
    Args:
        run_data: Raw diagnostic run data
        
    Returns:
        Tuple of (processed_data, validation_report)
    """
    validation_report = {
        "run_id": run_data.get("run_id", "unknown"),
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "anomalies": {}
    }
    
    # Validate features
    if "features" in run_data:
        feat_valid, feat_issues = validate_feature_quality(run_data["features"])
        if not feat_valid:
            validation_report["is_valid"] = False
            validation_report["issues"].extend(feat_issues)
        
        # Detect sensor anomalies
        anomalies = detect_sensor_anomalies(run_data["features"])
        if anomalies["has_anomalies"]:
            validation_report["warnings"].extend(anomalies["flags"])
            validation_report["anomalies"] = anomalies
    
    # Validate location
    if "fault_location" in run_data:
        loc_valid, loc_issues = validate_fault_location(run_data["fault_location"])
        if not loc_valid:
            validation_report["is_valid"] = False
            validation_report["issues"].extend(loc_issues)
    
    return run_data, validation_report


# Example usage in pipeline_runner.py
def run_pipeline_with_validation(input_runs: List[Dict[str, Any]], output_dir):
    """
    Enhanced pipeline with validation for real data.
    """
    validated_runs = []
    validation_reports = []
    
    for run_data in input_runs:
        processed_run, validation_report = preprocess_real_data(run_data)
        
        if validation_report["is_valid"]:
            validated_runs.append(processed_run)
            print(f"✓ {run_data['run_id']}: Valid")
            
            if validation_report["warnings"]:
                print(f"  ⚠ Warnings: {', '.join(validation_report['warnings'])}")
        else:
            print(f"✗ {run_data['run_id']}: Invalid")
            print(f"  Issues: {', '.join(validation_report['issues'])}")
        
        validation_reports.append(validation_report)
    
    # Run pipeline on validated data only
    if validated_runs:
        from pipeline_runner import run_person_c_pipeline
        results = run_person_c_pipeline(validated_runs, output_dir)
        return results, validation_reports
    else:
        return {}, validation_reports


if __name__ == "__main__":
    # Test with sample real data
    real_data = {
        "run_id": "REAL_001",
        "fault_location": {"x": 0.7, "y": 0.3},
        "features": {
            "dominant_frequency": 145.2,
            "rms_vibration": 3.8,
            "spectral_entropy": 0.78,
            "bearing_energy_band": 0.65,
            "audio_anomaly_score": 0.72,
            "speed_dependency": "strong"
        }
    }
    
    processed, report = preprocess_real_data(real_data)
    print("Validation Report:")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Issues: {report['issues']}")
    print(f"  Warnings: {report['warnings']}")
