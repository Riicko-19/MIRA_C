"""
Run Person C Pipeline on CWRU Bearing Dataset

Executes the complete Person C diagnostic pipeline on preprocessed CWRU data.
Does NOT provide ground_truth_cause as input - tests unsupervised diagnostic capability.

Pipeline stages:
1. Load preprocessed runs from data/processed/runs_features.jsonl
2. Run each through Person C pipeline (Fleet Matching → Causal Inference → Active Experiments → Scheduling → Explanation)
3. Save predictions to results/cwru/predictions.json
4. Save detailed outputs to results/cwru/person_c_output/

Usage:
    python experiments/run_person_c_on_dataset.py
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline_runner import run_person_c_pipeline


def load_preprocessed_runs(runs_file: Path) -> List[Dict[str, Any]]:
    """
    Load preprocessed diagnostic runs from JSONL file.
    
    Args:
        runs_file: Path to runs_features.jsonl
        
    Returns:
        List of diagnostic run dictionaries
    """
    if not runs_file.exists():
        raise FileNotFoundError(
            f"Preprocessed runs file not found: {runs_file}\n"
            "Please run: python datasets/preprocess_bearing_data.py"
        )
    
    runs = []
    with open(runs_file, 'r') as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    
    return runs


def prepare_runs_for_person_c(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare runs for Person C pipeline - REMOVE ground_truth_cause from input.
    
    Person C should diagnose faults unsupervised, without knowing the true cause.
    Ground truth is stored separately for evaluation.
    
    Args:
        runs: Raw runs with ground_truth in metadata
        
    Returns:
        Runs formatted for Person C input (without ground_truth)
    """
    person_c_runs = []
    
    for run in runs:
        # Extract ground_truth but don't pass it to Person C
        ground_truth = run["metadata"].pop("ground_truth_cause", "Unknown")
        
        # Keep only necessary fields for Person C input
        person_c_input = {
            "run_id": run["run_id"],
            "fault_location": run["fault_location"],
            "features": run["features"],
            "metadata": {
                "source": run["metadata"]["source"],
                "original_file": run["metadata"]["original_file"],
                "fault_code": run["metadata"]["fault_code"],
                "description": run["metadata"]["description"]
            }
        }
        
        person_c_runs.append(person_c_input)
    
    return person_c_runs


def extract_predictions(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract predictions from Person C pipeline results.
    
    Args:
        results: Full Person C pipeline results
        
    Returns:
        List of predictions with run_id, predicted_cause, confidence
    """
    predictions = []
    
    for run_id, result in results.items():
        if run_id == "_metadata":
            continue
        
        summary = result.get("summary", {})
        
        prediction = {
            "run_id": run_id,
            "predicted_cause": summary.get("root_cause", "Unknown"),
            "confidence": summary.get("confidence", 0.0),
            "requires_experiment": summary.get("requires_experiment", False),
            "repair_summary": summary.get("repair_summary", "")
        }
        
        predictions.append(prediction)
    
    return predictions


def main():
    """Main pipeline execution."""
    print("=" * 70)
    print("Person C Pipeline - CWRU Bearing Evaluation")
    print("=" * 70)
    
    # Paths
    processed_file = project_root / "data" / "processed" / "runs_features.jsonl"
    results_dir = project_root / "results" / "cwru"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = results_dir / "person_c_output"
    predictions_file = results_dir / "predictions.json"
    
    print(f"\n1. Input file: {processed_file}")
    print(f"2. Output directory: {output_dir}")
    print(f"3. Predictions file: {predictions_file}")
    
    # Load preprocessed runs
    print(f"\n4. Loading preprocessed runs...")
    try:
        raw_runs = load_preprocessed_runs(processed_file)
        print(f"   Loaded {len(raw_runs)} runs")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    # Store ground truth separately for evaluation
    ground_truth_map = {}
    for run in raw_runs:
        ground_truth_map[run["run_id"]] = run["metadata"]["ground_truth_cause"]
    
    # Prepare runs for Person C (WITHOUT ground_truth as input)
    print(f"\n5. Preparing runs for Person C pipeline...")
    person_c_runs = prepare_runs_for_person_c(raw_runs)
    print(f"   Removed ground_truth from inputs (will be used for evaluation only)")
    
    # Run Person C pipeline
    print(f"\n6. Running Person C diagnostic pipeline...")
    print("   This may take a minute - processing all 5 agents per run...")
    
    results = run_person_c_pipeline(
        person_c_runs,
        output_dir=output_dir,
        validate_inputs=True,  # Validate data quality
        confidence_threshold=0.75
    )
    
    # Extract predictions
    print(f"\n7. Extracting predictions...")
    predictions = extract_predictions(results)
    
    # Add ground truth back for evaluation
    for pred in predictions:
        pred["ground_truth"] = ground_truth_map.get(pred["run_id"], "Unknown")
    
    # Save predictions
    print(f"\n8. Saving predictions to {predictions_file.name}...")
    with open(predictions_file, 'w') as f:
        json.dump({
            "predictions": predictions,
            "metadata": {
                "dataset": "CWRU Bearing Dataset",
                "total_runs": len(predictions),
                "pipeline_version": "Person C v1.0"
            }
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Execution Complete!")
    print("=" * 70)
    
    print(f"\n📊 Processed {len(predictions)} runs:")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for pred in predictions[:5]:
        conf_pct = pred["confidence"] * 100
        print(f"  {pred['run_id']}: {pred['predicted_cause']} ({conf_pct:.1f}% confidence)")
        print(f"    Ground truth: {pred['ground_truth']}")
    
    if len(predictions) > 5:
        print(f"  ... and {len(predictions) - 5} more")
    
    # Quick accuracy preview
    correct = sum(1 for p in predictions if p["predicted_cause"] == p["ground_truth"])
    accuracy = correct / len(predictions) if predictions else 0
    print(f"\n📈 Quick accuracy: {correct}/{len(predictions)} = {accuracy*100:.1f}%")
    
    print(f"\n✓ Results saved to {results_dir}")
    print("  Next: python experiments/evaluate_person_c_on_dataset.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
