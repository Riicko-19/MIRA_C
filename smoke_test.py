"""
Smoke Test for MIRA Wave Person C - Phase 4

Generates random but physically plausible diagnostic runs to verify:
- No crashes or exceptions
- No NaN/inf in outputs
- Probabilities always sum to ~1.0
- Confidence values in [0, 1]
- All output files generated correctly

Run this before deployment to ensure system robustness.
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any
import sys

try:
    from pipeline_runner import run_person_c_pipeline
    from utils import convert_to_serializable
except ImportError:
    print("Error: Could not import pipeline_runner or utils. Run from project root.")
    sys.exit(1)


def generate_random_runs(n_runs: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate N random but physically plausible diagnostic runs.
    
    Args:
        n_runs: Number of runs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of diagnostic run dictionaries
    """
    np.random.seed(seed)
    runs = []
    
    for i in range(n_runs):
        # Generate realistic feature values
        run = {
            "run_id": f"SMOKE_{i:04d}",
            "fault_location": {
                "x": float(np.random.uniform(0, 1)),
                "y": float(np.random.uniform(0, 1))
            },
            "features": {
                # Frequency: log-uniform in automotive range (10-4000 Hz)
                "dominant_frequency": float(np.exp(np.random.uniform(np.log(10), np.log(4000)))),
                
                # Vibration: mostly 0.5-10 m/s², occasionally higher
                "rms_vibration": float(np.random.gamma(2, 2) if np.random.rand() < 0.9 
                                      else np.random.uniform(10, 15)),
                
                # Entropy: mostly 0.3-0.9, occasionally extreme
                "spectral_entropy": float(np.clip(np.random.beta(5, 2), 0, 1)),
                
                # Bearing energy: uniform 0-1
                "bearing_energy_band": float(np.random.uniform(0, 1)),
                
                # Audio score: beta distribution favoring mid-range
                "audio_anomaly_score": float(np.random.beta(2, 2)),
                
                # Speed dependency: weighted random choice
                "speed_dependency": np.random.choice(
                    ["weak", "medium", "strong"],
                    p=[0.2, 0.3, 0.5]
                )
            },
            "metadata": {
                "test_type": "smoke_test",
                "run_index": i
            }
        }
        
        runs.append(run)
    
    return runs


def validate_output_file(filepath: Path) -> Dict[str, Any]:
    """
    Validate a JSON output file for correctness.
    
    Returns:
        Dict with validation results
    """
    results = {
        "exists": filepath.exists(),
        "valid_json": False,
        "has_nan": False,
        "has_inf": False,
        "probabilities_valid": True,
        "errors": []
    }
    
    if not results["exists"]:
        results["errors"].append(f"File does not exist: {filepath}")
        return results
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        results["valid_json"] = True
        
        # Check for NaN/inf in JSON (they become null or invalid strings)
        json_str = json.dumps(data)
        if "NaN" in json_str or "Infinity" in json_str or "-Infinity" in json_str:
            results["has_nan"] = True
            results["errors"].append("Contains NaN or Infinity values")
        
        # Validate probabilities if it's a cause file
        if "confidence" in data:
            confidence = data["confidence"]
            if not (0 <= confidence <= 1):
                results["probabilities_valid"] = False
                results["errors"].append(f"Confidence {confidence} out of range [0, 1]")
            
            if "alternate_causes" in data:
                probs = [confidence] + list(data["alternate_causes"].values())
                prob_sum = sum(probs)
                if not (0.95 <= prob_sum <= 1.05):  # Allow small floating point error
                    results["probabilities_valid"] = False
                    results["errors"].append(f"Probabilities sum to {prob_sum}, not ~1.0")
        
    except json.JSONDecodeError as e:
        results["errors"].append(f"Invalid JSON: {e}")
    except Exception as e:
        results["errors"].append(f"Validation error: {e}")
    
    return results


def run_smoke_test(n_runs: int = 100, verbose: bool = True):
    """
    Run comprehensive smoke test.
    
    Args:
        n_runs: Number of random runs to test
        verbose: Print detailed progress
    """
    print("=" * 70)
    print("MIRA Wave Person C - SMOKE TEST (Phase 4)")
    print("=" * 70)
    print(f"\nGenerating {n_runs} random diagnostic runs...")
    
    # Generate random runs
    runs = generate_random_runs(n_runs)
    
    print(f"✓ Generated {len(runs)} runs with realistic feature distributions\n")
    
    # Run pipeline
    output_dir = Path("./smoke_test_output")
    print(f"Running pipeline (output: {output_dir})...")
    
    try:
        results = run_person_c_pipeline(
            runs,
            output_dir=output_dir,
            validate_inputs=True,
            confidence_threshold=0.75
        )
        print("✓ Pipeline completed without exceptions\n")
    except Exception as e:
        print(f"✗ CRITICAL: Pipeline crashed with exception:")
        print(f"  {type(e).__name__}: {e}")
        return False
    
    # Analyze results
    metadata = results.get("_metadata", {})
    validation_summary = metadata.get("validation_summary", {})
    
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Runs:       {validation_summary.get('total', 0)}")
    print(f"Valid Runs:       {validation_summary.get('valid', 0)}")
    print(f"Invalid Runs:     {validation_summary.get('invalid', 0)}")
    print(f"Warnings Issued:  {validation_summary.get('warnings', 0)}")
    
    valid_runs = validation_summary.get('valid', 0)
    
    if valid_runs == 0:
        print("\n✗ No valid runs to process!")
        return False
    
    # Check each processed run
    print(f"\n{'=' * 70}")
    print("OUTPUT VALIDATION")
    print("=" * 70)
    
    issues_found = 0
    runs_checked = 0
    
    for run_id, result in results.items():
        if run_id == "_metadata":
            continue
        
        runs_checked += 1
        
        # Check for NaN/inf in result structure (more precise check)
        # Convert to JSON string and check for actual NaN/Infinity values
        try:
            # Use our safe serialization helper
            serializable_result = convert_to_serializable(result)
            result_json = json.dumps(serializable_result)
            # Check for JavaScript NaN/Infinity representations
            if ": NaN" in result_json or ": Infinity" in result_json or ": -Infinity" in result_json:
                print(f"✗ {run_id}: Contains NaN/Infinity numeric values")
                issues_found += 1
                continue
        except (TypeError, ValueError) as e:
            print(f"✗ {run_id}: Cannot serialize to JSON: {e}")
            issues_found += 1
            continue
        
        # Check confidence value
        summary = result.get("summary", {})
        confidence = summary.get("confidence", -1)
        
        if not (0 <= confidence <= 1):
            print(f"✗ {run_id}: Invalid confidence {confidence}")
            issues_found += 1
        
        # Validate output files
        cause_file = output_dir / f"cause_{run_id}.json"
        experiment_file = output_dir / f"experiment_{run_id}.json"
        repair_file = output_dir / f"repair_{run_id}.json"
        explanation_file = output_dir / f"explanation_{run_id}.txt"
        
        for filepath in [cause_file, experiment_file, repair_file]:
            validation = validate_output_file(filepath)
            
            if not validation["exists"]:
                print(f"✗ {run_id}: Missing {filepath.name}")
                issues_found += 1
            elif not validation["valid_json"]:
                print(f"✗ {run_id}: Invalid JSON in {filepath.name}")
                issues_found += 1
            elif validation["has_nan"]:
                print(f"✗ {run_id}: NaN/Inf in {filepath.name}")
                issues_found += 1
            elif not validation["probabilities_valid"]:
                print(f"✗ {run_id}: Invalid probabilities in {filepath.name}")
                for error in validation["errors"]:
                    print(f"    {error}")
                issues_found += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    print(f"Runs Checked:     {runs_checked}")
    print(f"Issues Found:     {issues_found}")
    
    if issues_found == 0:
        print(f"\n✓ ALL CHECKS PASSED - System is robust!")
        print(f"  - No crashes or exceptions")
        print(f"  - No NaN or Infinity values")
        print(f"  - All probabilities in valid range")
        print(f"  - All output files generated correctly")
        return True
    else:
        print(f"\n✗ ISSUES DETECTED - Review failures above")
        return False


def quick_smoke_test():
    """Quick smoke test with 20 runs for rapid verification."""
    print("Running QUICK smoke test (20 runs)...\n")
    return run_smoke_test(n_runs=20, verbose=True)


def full_smoke_test():
    """Full smoke test with 100 runs for comprehensive verification."""
    print("Running FULL smoke test (100 runs)...\n")
    return run_smoke_test(n_runs=100, verbose=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        success = full_smoke_test()
    else:
        success = quick_smoke_test()
    
    # Exit with proper code
    sys.exit(0 if success else 1)
