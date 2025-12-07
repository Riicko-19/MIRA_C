"""
Example demonstrating data validation for real sensor data.

Shows how the system handles invalid inputs and warns about data quality issues.
"""

from pathlib import Path
from pipeline_runner import run_person_c_pipeline


# Sample runs with various data quality issues
test_runs = [
    # Valid run
    {
        "run_id": "VALID_001",
        "fault_location": {"x": 0.65, "y": 0.32},
        "features": {
            "dominant_frequency": 145.2,
            "rms_vibration": 3.8,
            "spectral_entropy": 0.78,
            "bearing_energy_band": 0.65,
            "audio_anomaly_score": 0.72,
            "speed_dependency": "strong"
        },
        "metadata": {"vehicle_id": "TEST_VALID", "sim_or_real": "real"}
    },
    
    # Invalid: Frequency out of range
    {
        "run_id": "INVALID_FREQ",
        "fault_location": {"x": 0.5, "y": 0.5},
        "features": {
            "dominant_frequency": 6000,  # Too high!
            "rms_vibration": 3.5,
            "spectral_entropy": 0.75,
            "bearing_energy_band": 0.60,
            "audio_anomaly_score": 0.70,
            "speed_dependency": "medium"
        }
    },
    
    # Warning: Possible sensor saturation
    {
        "run_id": "WARNING_SAT",
        "fault_location": {"x": 0.7, "y": 0.3},
        "features": {
            "dominant_frequency": 150.0,
            "rms_vibration": 12.5,  # Very high - possible saturation
            "spectral_entropy": 0.80,
            "bearing_energy_band": 0.65,
            "audio_anomaly_score": 0.75,
            "speed_dependency": "strong"
        }
    },
    
    # Invalid: Missing required fields
    {
        "run_id": "INVALID_MISSING",
        "fault_location": {"x": 0.5, "y": 0.5},
        "features": {
            "dominant_frequency": 120.0,
            "rms_vibration": 3.0
            # Missing other required features
        }
    },
    
    # Warning: Unrealistic combination
    {
        "run_id": "WARNING_COMBO",
        "fault_location": {"x": 0.4, "y": 0.4},
        "features": {
            "dominant_frequency": 80.0,  # Low frequency
            "rms_vibration": 3.2,
            "spectral_entropy": 0.72,
            "bearing_energy_band": 0.95,  # High bearing energy - unusual combo
            "audio_anomaly_score": 0.68,
            "speed_dependency": "weak"
        }
    },
    
    # Invalid: Entropy out of range
    {
        "run_id": "INVALID_ENTROPY",
        "fault_location": {"x": 0.6, "y": 0.4},
        "features": {
            "dominant_frequency": 200.0,
            "rms_vibration": 3.5,
            "spectral_entropy": 1.5,  # > 1.0 is invalid
            "bearing_energy_band": 0.60,
            "audio_anomaly_score": 0.70,
            "speed_dependency": "medium"
        }
    },
    
    # Valid but with warnings - low entropy
    {
        "run_id": "WARNING_LOW_ENT",
        "fault_location": {"x": 0.3, "y": 0.6},
        "features": {
            "dominant_frequency": 95.0,
            "rms_vibration": 2.8,
            "spectral_entropy": 0.03,  # Very low - possible sensor issue
            "bearing_energy_band": 0.40,
            "audio_anomaly_score": 0.55,
            "speed_dependency": "strong"
        }
    }
]


def main():
    """Run validation test."""
    
    print("=" * 80)
    print("MIRA Wave Person C - Data Validation Demo")
    print("=" * 80)
    print()
    print(f"Testing {len(test_runs)} diagnostic runs with various data quality issues...")
    print()
    
    output_dir = Path("./validation_test_output")
    
    # Run with validation enabled
    results = run_person_c_pipeline(
        test_runs,
        output_dir=output_dir,
        validate_inputs=True,  # Enable validation
        confidence_threshold=0.75
    )
    
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    if "_metadata" in results:
        metadata = results["_metadata"]
        val_sum = metadata.get("validation_summary", {})
        
        print(f"Total Runs Submitted: {val_sum['total']}")
        print(f"Valid Runs: {val_sum['valid']}")
        print(f"Invalid Runs: {val_sum['invalid']}")
        print(f"Total Warnings: {val_sum['warnings']}")
        print()
        
        print(f"Successfully Processed: {metadata['total_runs_processed']} run(s)")
        print()
    
    # Show which runs were processed
    processed_runs = [rid for rid in results.keys() if rid != "_metadata"]
    if processed_runs:
        print("Processed Runs:")
        for run_id in processed_runs:
            summary = results[run_id]["summary"]
            print(f"  ✓ {run_id}: {summary['root_cause']} ({summary['confidence']*100:.1f}% confidence)")
    
    print()
    print("=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("1. Invalid data is rejected before processing")
    print("2. Warnings are shown for suspicious but valid data")
    print("3. Only validated runs proceed through the diagnostic pipeline")
    print("4. This prevents garbage-in-garbage-out scenarios")
    print()


if __name__ == "__main__":
    main()
