"""
Example usage of MIRA Wave Person C diagnostic system.

Demonstrates how to run the complete multi-agent pipeline on sample diagnostic data.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_runner import run_person_c_pipeline

# Sample diagnostic runs
sample_runs = [
    {
        "run_id": "R001",
        "fault_location": {
            "x": 0.72,
            "y": 0.31
        },
        "features": {
            "dominant_frequency": 132.4,
            "rms_vibration": 3.6,
            "spectral_entropy": 0.81,
            "bearing_energy_band": 0.67,
            "audio_anomaly_score": 0.74,
            "speed_dependency": "strong"
        },
        "metadata": {
            "vehicle_id": "V_102",
            "sim_or_real": "simulated",
            "severity_hint": "medium"
        }
    },
    {
        "run_id": "R002",
        "fault_location": {
            "x": 0.25,
            "y": 0.15
        },
        "features": {
            "dominant_frequency": 2847.0,
            "rms_vibration": 2.3,
            "spectral_entropy": 0.93,
            "bearing_energy_band": 0.89,
            "audio_anomaly_score": 0.82,
            "speed_dependency": "weak"
        },
        "metadata": {
            "vehicle_id": "V_103",
            "sim_or_real": "simulated",
            "severity_hint": "high"
        }
    },
    {
        "run_id": "R003",
        "fault_location": {
            "x": 0.15,
            "y": 0.45
        },
        "features": {
            "dominant_frequency": 67.2,
            "rms_vibration": 3.1,
            "spectral_entropy": 0.52,
            "bearing_energy_band": 0.35,
            "audio_anomaly_score": 0.61,
            "speed_dependency": "strong"
        },
        "metadata": {
            "vehicle_id": "V_104",
            "sim_or_real": "simulated",
            "severity_hint": "low"
        }
    }
]


def main():
    """Run the complete diagnostic pipeline on sample data."""
    
    print("=" * 80)
    print("MIRA Wave Person C - Multi-Agent Diagnostic System")
    print("UPGRADED FOR REAL DATA COMPATIBILITY")
    print("=" * 80)
    print()
    
    # Set output directory
    output_dir = Path("./output")
    
    # Optional: Path to real fleet database (if available)
    fleet_db_path = Path("./real_fleet_history.json")
    
    # Run pipeline with validation enabled and optional real fleet
    print(f"Processing {len(sample_runs)} diagnostic runs...")
    print()
    
    results = run_person_c_pipeline(
        sample_runs, 
        output_dir=output_dir,
        fleet_database_path=fleet_db_path if fleet_db_path.exists() else None,
        validate_inputs=True,  # Enable validation for real data
        confidence_threshold=0.75  # Adjustable threshold
    )
    
    # Display summary results
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print()
    
    # Show metadata if available
    if "_metadata" in results:
        metadata = results["_metadata"]
        print(f"Runs Processed: {metadata['total_runs_processed']}")
        print(f"Fleet Source: {metadata['fleet_source']}")
        print(f"Confidence Threshold: {metadata['confidence_threshold']}")
        
        if metadata.get("validation_summary"):
            val_sum = metadata["validation_summary"]
            if val_sum["warnings"] > 0:
                print(f"Warnings: {val_sum['warnings']}")
        print()
        print("-" * 80)
        print()
    
    for run_id, result in results.items():
        if run_id == "_metadata":  # Skip metadata in iteration
            continue
            
        summary = result.get("summary", {})
        
        print(f"Run ID: {run_id}")
        print(f"  Root Cause: {summary.get('root_cause', 'Unknown')}")
        print(f"  Confidence: {summary.get('confidence', 0)*100:.1f}%")
        print(f"  Urgency: {summary.get('urgency', 'Unknown')}")
        print(f"  Experiment Required: {'Yes' if summary.get('experiment_required') else 'No'}")
        print(f"  Estimated Downtime: {summary.get('estimated_downtime_hours', 0)} hours")
        print()
    
    print("=" * 80)
    print(f"Output files saved to: {output_dir.absolute()}")
    print("=" * 80)
    print()
    print("Generated files for each run:")
    print("  - cause_<run_id>.json         : Causal diagnosis with probabilities")
    print("  - experiment_<run_id>.json    : Active experiment protocol")
    print("  - repair_<run_id>.json        : Repair plan with urgency and steps")
    print("  - explanation_<run_id>.txt    : Human-readable report")
    print("  - explanation_<run_id>.json   : Structured explanation")
    print()


if __name__ == "__main__":
    main()
