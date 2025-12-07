"""
Evaluate Person C Performance on CWRU Bearing Dataset (UPGRADED)

Computes comprehensive evaluation metrics comparing Person C predictions
against ground truth labels from CWRU dataset.

UPGRADES (Phase C):
- Per-class metrics with sub-type granularity (bearing_ball, bearing_inner, bearing_outer, normal)
- Expected Calibration Error (ECE) with reliability curves
- Fault detection vs fault classification accuracy (2-stage metrics)
- Detailed error analysis with feature vectors and explanations
- Macro-F1 score computation

Metrics computed:
- Confusion matrix (sub-type level)
- Overall accuracy
- Fault detection accuracy (normal vs fault)
- Fault classification accuracy (conditional on being faulty)
- Per-class precision, recall, F1-score, macro-F1
- Confidence calibration (ECE, reliability curves)
- Error analysis with feature logging to results/cwru/error_analysis.json

Usage:
    python experiments/evaluate_person_c_on_dataset.py
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_predictions(predictions_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load predictions from JSON file.
    
    UPGRADE: Also loads features and sub-type labels from runs_features.jsonl
    
    Args:
        predictions_file: Path to predictions.json
        
    Returns:
        (predictions_list, metadata)
    """
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            "Please run: python experiments/run_person_c_on_dataset.py"
        )
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = data["predictions"]
    metadata = data["metadata"]
    
    # UPGRADE: Load sub-type labels and features from runs_features.jsonl
    runs_file = project_root / "data" / "processed" / "runs_features.jsonl"
    if runs_file.exists():
        runs_by_id = {}
        with open(runs_file, 'r') as f:
            for line in f:
                if line.strip():
                    run = json.loads(line)
                    runs_by_id[run["run_id"]] = run
        
        # Enrich predictions with sub-type and features
        for pred in predictions:
            run_id = pred["run_id"]
            if run_id in runs_by_id:
                run_data = runs_by_id[run_id]
                pred["ground_truth_subtype"] = run_data["metadata"].get("ground_truth_subtype", "unknown")
                pred["features"] = run_data["features"]
                pred["metadata"] = run_data["metadata"]
    
    return predictions, metadata


def compute_confusion_matrix(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix from predictions.
    
    Returns:
        Nested dict: confusion[true_label][predicted_label] = count
    """
    confusion = defaultdict(lambda: defaultdict(int))
    
    for pred in predictions:
        true_label = pred["ground_truth"]
        pred_label = pred["predicted_cause"]
        confusion[true_label][pred_label] += 1
    
    return dict(confusion)


def compute_accuracy(predictions: List[Dict[str, Any]]) -> float:
    """
    Compute overall accuracy.
    
    Args:
        predictions: List of predictions with ground_truth and predicted_cause
        
    Returns:
        Accuracy as fraction [0, 1]
    """
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p["predicted_cause"] == p["ground_truth"])
    return correct / len(predictions)


def compute_per_class_metrics(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1-score per class.
    
    Args:
        confusion: Confusion matrix
        
    Returns:
        Dict mapping class -> {precision, recall, f1, support}
    """
    metrics = {}
    
    # Get all classes
    all_classes = set(confusion.keys())
    for pred_dict in confusion.values():
        all_classes.update(pred_dict.keys())
    
    for cls in all_classes:
        # True positives: confusion[cls][cls]
        tp = confusion.get(cls, {}).get(cls, 0)
        
        # False positives: sum of confusion[other][cls] for all other != cls
        fp = sum(confusion.get(other, {}).get(cls, 0) for other in all_classes if other != cls)
        
        # False negatives: sum of confusion[cls][other] for all other != cls
        fn = sum(confusion.get(cls, {}).get(other, 0) for other in all_classes if other != cls)
        
        # True negatives: not needed for precision/recall
        
        # Support: total ground truth instances of this class
        support = sum(confusion.get(cls, {}).values())
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    
    return metrics


def analyze_confidence_distribution(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze confidence score distribution.
    
    Args:
        predictions: List of predictions
        
    Returns:
        Dict with confidence statistics
    """
    confidences = [p["confidence"] for p in predictions]
    
    if not confidences:
        return {}
    
    # Separate correct vs incorrect predictions
    correct_confidences = [p["confidence"] for p in predictions if p["predicted_cause"] == p["ground_truth"]]
    incorrect_confidences = [p["confidence"] for p in predictions if p["predicted_cause"] != p["ground_truth"]]
    
    stats = {
        "overall": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences))
        }
    }
    
    if correct_confidences:
        stats["correct"] = {
            "mean": float(np.mean(correct_confidences)),
            "std": float(np.std(correct_confidences)),
            "count": len(correct_confidences)
        }
    
    if incorrect_confidences:
        stats["incorrect"] = {
            "mean": float(np.mean(incorrect_confidences)),
            "std": float(np.std(incorrect_confidences)),
            "count": len(incorrect_confidences)
        }
    
    return stats


def analyze_errors(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detailed error analysis.
    
    Args:
        predictions: List of predictions
        
    Returns:
        List of error cases with details
    """
    errors = []
    
    for pred in predictions:
        if pred["predicted_cause"] != pred["ground_truth"]:
            error = {
                "run_id": pred["run_id"],
                "true_label": pred["ground_truth"],
                "predicted_label": pred["predicted_cause"],
                "confidence": pred["confidence"],
                "requires_experiment": pred.get("requires_experiment", False)
            }
            errors.append(error)
    
    return errors


def print_confusion_matrix(confusion: Dict[str, Dict[str, int]], all_classes: List[str]):
    """
    Pretty print confusion matrix.
    
    Args:
        confusion: Confusion matrix
        all_classes: List of all class labels
    """
    print("\n" + "=" * 70)
    print("Confusion Matrix")
    print("=" * 70)
    print("(Rows = Ground Truth, Columns = Predicted)\n")
    
    # Header
    col_width = 18
    print(" " * col_width, end="")
    for pred_cls in sorted(all_classes):
        print(f"{pred_cls[:16]:>18}", end="")
    print()
    print("-" * (col_width + len(all_classes) * 18))
    
    # Rows
    for true_cls in sorted(all_classes):
        print(f"{true_cls[:16]:<{col_width}}", end="")
        for pred_cls in sorted(all_classes):
            count = confusion.get(true_cls, {}).get(pred_cls, 0)
            print(f"{count:>18}", end="")
        print()


def print_classification_report(metrics: Dict[str, Dict[str, float]]):
    """
    Pretty print classification report.
    
    Args:
        metrics: Per-class metrics
    """
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 70)
    
    for cls in sorted(metrics.keys()):
        m = metrics[cls]
        print(f"{cls:<20} {m['precision']:>12.3f} {m['recall']:>12.3f} {m['f1']:>12.3f} {m['support']:>12}")
    
    # Compute averages
    total_support = sum(m["support"] for m in metrics.values())
    if total_support > 0:
        avg_precision = sum(m["precision"] * m["support"] for m in metrics.values()) / total_support
        avg_recall = sum(m["recall"] * m["support"] for m in metrics.values()) / total_support
        avg_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / total_support
        
        print("-" * 70)
        print(f"{'Weighted Average':<20} {avg_precision:>12.3f} {avg_recall:>12.3f} {avg_f1:>12.3f} {total_support:>12}")


def print_confidence_analysis(conf_stats: Dict[str, Any]):
    """
    Pretty print confidence distribution analysis.
    
    Args:
        conf_stats: Confidence statistics
    """
    print("\n" + "=" * 70)
    print("Confidence Distribution Analysis")
    print("=" * 70)
    
    overall = conf_stats.get("overall", {})
    print(f"\nOverall Confidence:")
    print(f"  Mean:   {overall.get('mean', 0):.3f}")
    print(f"  Median: {overall.get('median', 0):.3f}")
    print(f"  Std:    {overall.get('std', 0):.3f}")
    print(f"  Range:  [{overall.get('min', 0):.3f}, {overall.get('max', 0):.3f}]")
    
    if "correct" in conf_stats:
        correct = conf_stats["correct"]
        print(f"\nCorrect Predictions ({correct['count']} cases):")
        print(f"  Mean confidence: {correct['mean']:.3f} ± {correct['std']:.3f}")
    
    if "incorrect" in conf_stats:
        incorrect = conf_stats["incorrect"]
        print(f"\nIncorrect Predictions ({incorrect['count']} cases):")
        print(f"  Mean confidence: {incorrect['mean']:.3f} ± {incorrect['std']:.3f}")


def print_error_analysis(errors: List[Dict[str, Any]]):
    """
    Pretty print error analysis.
    
    Args:
        errors: List of error cases
    """
    print("\n" + "=" * 70)
    print("Error Analysis")
    print("=" * 70)
    
    if not errors:
        print("\n✓ No errors - perfect classification!")
        return
    
    print(f"\nTotal Errors: {len(errors)}\n")
    
    # Group errors by true label
    errors_by_true = defaultdict(list)
    for err in errors:
        errors_by_true[err["true_label"]].append(err)
    
    for true_label in sorted(errors_by_true.keys()):
        errs = errors_by_true[true_label]
        print(f"\n{true_label} (misclassified {len(errs)} time(s)):")
        
        # Count misclassifications by predicted label
        pred_counts = Counter(e["predicted_label"] for e in errs)
        for pred_label, count in pred_counts.most_common():
            avg_conf = np.mean([e["confidence"] for e in errs if e["predicted_label"] == pred_label])
            print(f"  → Predicted as {pred_label}: {count} time(s) (avg conf: {avg_conf:.3f})")
        
        # Show specific examples
        for err in errs[:3]:  # Show first 3 examples per class
            print(f"    • {err['run_id']}: {err['true_label']} → {err['predicted_label']} ({err['confidence']:.3f})")


def save_evaluation_report(results_dir: Path, accuracy: float, confusion: Dict, metrics: Dict, conf_stats: Dict, errors: List):
    """
    Save comprehensive evaluation report to JSON.
    
    Args:
        results_dir: Directory to save report
        accuracy: Overall accuracy
        confusion: Confusion matrix
        metrics: Per-class metrics
        conf_stats: Confidence statistics
        errors: List of errors
    """
    report_file = results_dir / "evaluation_report.json"
    
    report = {
        "summary": {
            "overall_accuracy": accuracy,
            "total_predictions": sum(sum(row.values()) for row in confusion.values()),
            "total_errors": len(errors)
        },
        "confusion_matrix": confusion,
        "per_class_metrics": metrics,
        "confidence_statistics": conf_stats,
        "errors": errors
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Detailed evaluation report saved to {report_file.name}")


# ============================================================================
# UPGRADE: New Phase C Functions
# ============================================================================

def compute_subtype_confusion_matrix(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    UPGRADE: Compute confusion matrix at sub-type granularity.
    
    Uses ground_truth_subtype (bearing_ball, bearing_inner, bearing_outer, normal)
    vs predicted_cause (mapped to sub-types).
    
    Returns:
        Nested dict: confusion[true_subtype][predicted_subtype] = count
    """
    confusion = defaultdict(lambda: defaultdict(int))
    
    # Map Person C labels to sub-types (for predicted)
    bearing_to_generic = {
        "Bearing Wear": "bearing_generic",  # Person C doesn't distinguish sub-types
        "Normal": "normal",
        "Loose Mount": "loose_mount",
        "Imbalance": "imbalance",
        "Misalignment": "misalignment"
    }
    
    for pred in predictions:
        true_subtype = pred.get("ground_truth_subtype", "unknown")
        pred_label = pred["predicted_cause"]
        pred_subtype = bearing_to_generic.get(pred_label, "unknown")
        
        confusion[true_subtype][pred_subtype] += 1
    
    return dict(confusion)


def compute_fault_detection_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    UPGRADE: Compute 2-stage metrics:
    1. Fault detection: Normal vs Faulty
    2. Fault classification: Type of fault (conditional on being faulty)
    
    Returns:
        Dict with detection and classification accuracies
    """
    # Stage 1: Fault detection (binary classification)
    detection_correct = 0
    detection_total = 0
    
    # Stage 2: Fault classification (multi-class, only for faulty cases)
    classification_correct = 0
    classification_total = 0
    
    for pred in predictions:
        true_label = pred["ground_truth"]
        pred_label = pred["predicted_cause"]
        true_subtype = pred.get("ground_truth_subtype", "unknown")
        
        # Is it truly faulty?
        true_is_faulty = (true_subtype != "normal")
        pred_is_faulty = (pred_label != "Normal")
        
        # Detection accuracy
        if true_is_faulty == pred_is_faulty:
            detection_correct += 1
        detection_total += 1
        
        # Classification accuracy (only count faulty cases)
        if true_is_faulty and pred_is_faulty:
            # Both agree it's faulty - check if type is correct
            # For bearing sub-types, Person C predicts "Bearing Wear" (correct if any bearing subtype)
            if pred_label == "Bearing Wear" and true_subtype in ["bearing_ball", "bearing_inner", "bearing_outer"]:
                classification_correct += 1
            elif pred_label == true_label:  # Other fault types
                classification_correct += 1
            classification_total += 1
    
    detection_accuracy = detection_correct / detection_total if detection_total > 0 else 0.0
    classification_accuracy = classification_correct / classification_total if classification_total > 0 else 0.0
    
    return {
        "fault_detection_accuracy": detection_accuracy,
        "fault_classification_accuracy": classification_accuracy,
        "detection_correct": detection_correct,
        "detection_total": detection_total,
        "classification_correct": classification_correct,
        "classification_total": classification_total
    }


def compute_expected_calibration_error(predictions: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """
    UPGRADE: Compute Expected Calibration Error (ECE) and reliability curve.
    
    ECE measures how well predicted confidences match actual accuracy.
    
    Args:
        predictions: List of predictions with confidence scores
        n_bins: Number of bins for reliability curve (default 10)
        
    Returns:
        Dict with ECE, reliability curve, and per-bin statistics
    """
    # Bin predictions by confidence
    bins = [[] for _ in range(n_bins)]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for pred in predictions:
        conf = pred["confidence"]
        is_correct = (pred["predicted_cause"] == pred["ground_truth"])
        
        # Find bin
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, is_correct))
    
    # Compute per-bin statistics
    reliability_curve = []
    ece = 0.0
    total_samples = len(predictions)
    
    for bin_idx, bin_data in enumerate(bins):
        if not bin_data:
            continue
        
        confidences = [conf for conf, _ in bin_data]
        correctness = [int(correct) for _, correct in bin_data]
        
        avg_confidence = np.mean(confidences)
        avg_accuracy = np.mean(correctness)
        bin_size = len(bin_data)
        
        # ECE contribution: weighted absolute difference
        ece += (bin_size / total_samples) * abs(avg_confidence - avg_accuracy)
        
        reliability_curve.append({
            "bin_index": bin_idx,
            "bin_range": [float(bin_edges[bin_idx]), float(bin_edges[bin_idx + 1])],
            "avg_confidence": float(avg_confidence),
            "avg_accuracy": float(avg_accuracy),
            "count": bin_size,
            "gap": float(avg_confidence - avg_accuracy)
        })
    
    return {
        "expected_calibration_error": float(ece),
        "n_bins": n_bins,
        "reliability_curve": reliability_curve
    }


def compute_macro_f1(metrics: Dict[str, Dict[str, float]]) -> float:
    """
    UPGRADE: Compute macro-F1 score (unweighted average of per-class F1).
    
    Args:
        metrics: Per-class metrics dict
        
    Returns:
        Macro-F1 score
    """
    f1_scores = [m["f1"] for m in metrics.values()]
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def analyze_errors_with_features(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    UPGRADE: Enhanced error analysis with feature vectors and explanations.
    
    Args:
        predictions: List of predictions with features
        
    Returns:
        List of error dicts with features, confidence, and reasoning
    """
    errors = []
    
    for pred in predictions:
        if pred["predicted_cause"] != pred["ground_truth"]:
            error = {
                "run_id": pred["run_id"],
                "true_label": pred["ground_truth"],
                "true_subtype": pred.get("ground_truth_subtype", "unknown"),
                "predicted_label": pred["predicted_cause"],
                "confidence": pred["confidence"],
                "requires_experiment": pred.get("requires_experiment", False)
            }
            
            # Add features if available
            if "features" in pred:
                error["features"] = pred["features"]
            
            # Add metadata if available
            if "metadata" in pred:
                error["original_file"] = pred["metadata"].get("original_file", "unknown")
                error["fault_code"] = pred["metadata"].get("fault_code", "unknown")
                error["description"] = pred["metadata"].get("description", "unknown")
            
            # Build explanation
            if "features" in pred:
                f = pred["features"]
                explanation_parts = []
                explanation_parts.append(f"RMS={f['rms_vibration']:.3f}")
                explanation_parts.append(f"freq={f['dominant_frequency']:.1f}Hz")
                explanation_parts.append(f"entropy={f['spectral_entropy']:.3f}")
                explanation_parts.append(f"bearing_energy={f['bearing_energy_band']:.3f}")
                explanation_parts.append(f"anomaly={f['audio_anomaly_score']:.3f}")
                error["explanation"] = ", ".join(explanation_parts)
            else:
                error["explanation"] = "Features not available"
            
            errors.append(error)
    
    return errors


def save_error_analysis_json(results_dir: Path, errors: List[Dict[str, Any]]):
    """
    UPGRADE: Save detailed error analysis to error_analysis.json.
    
    Args:
        results_dir: Directory to save error analysis
        errors: List of error dicts with features
    """
    error_file = results_dir / "error_analysis.json"
    
    error_report = {
        "total_errors": len(errors),
        "errors": errors
    }
    
    with open(error_file, 'w') as f:
        json.dump(error_report, f, indent=2)
    
    print(f"✓ Detailed error analysis saved to {error_file.name}")


def save_calibration_json(results_dir: Path, calibration: Dict[str, Any]):
    """
    UPGRADE: Save calibration analysis to calibration.json.
    
    Args:
        results_dir: Directory to save calibration
        calibration: Calibration dict with ECE and reliability curve
    """
    calib_file = results_dir / "calibration.json"
    
    with open(calib_file, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"✓ Calibration analysis saved to {calib_file.name}")


# ============================================================================
# End of UPGRADE functions
# ============================================================================


def main():
    """Main evaluation pipeline (UPGRADED)."""
    print("=" * 70)
    print("Person C Evaluation - CWRU Bearing Dataset (UPGRADED)")
    print("=" * 70)
    
    # Paths
    results_dir = project_root / "results" / "cwru"
    predictions_file = results_dir / "predictions.json"
    
    print(f"\n1. Loading predictions from {predictions_file.name}...")
    try:
        predictions, metadata = load_predictions(predictions_file)
        print(f"   Loaded {len(predictions)} predictions")
        print(f"   Dataset: {metadata.get('dataset', 'Unknown')}")
        print(f"   Sub-type labels: {len([p for p in predictions if 'ground_truth_subtype' in p])} enriched")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    # Compute metrics
    print(f"\n2. Computing evaluation metrics...")
    
    # Standard metrics
    accuracy = compute_accuracy(predictions)
    confusion = compute_confusion_matrix(predictions)
    
    # Get all classes
    all_classes = set()
    for p in predictions:
        all_classes.add(p["ground_truth"])
        all_classes.add(p["predicted_cause"])
    
    metrics = compute_per_class_metrics(confusion)
    conf_stats = analyze_confidence_distribution(predictions)
    
    # UPGRADE: New metrics
    print(f"   Computing sub-type confusion matrix...")
    subtype_confusion = compute_subtype_confusion_matrix(predictions)
    
    print(f"   Computing fault detection metrics...")
    detection_metrics = compute_fault_detection_metrics(predictions)
    
    print(f"   Computing calibration (ECE)...")
    calibration = compute_expected_calibration_error(predictions, n_bins=10)
    
    print(f"   Computing macro-F1...")
    macro_f1 = compute_macro_f1(metrics)
    
    print(f"   Analyzing errors with features...")
    errors_enhanced = analyze_errors_with_features(predictions)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (UPGRADED)")
    print("=" * 70)
    
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}% ({int(accuracy*len(predictions))}/{len(predictions)} correct)")
    
    # UPGRADE: 2-stage metrics
    print(f"\n📊 Fault Detection & Classification:")
    print(f"   Fault Detection Accuracy: {detection_metrics['fault_detection_accuracy']*100:.2f}% "
          f"({detection_metrics['detection_correct']}/{detection_metrics['detection_total']} correct)")
    print(f"   Fault Classification Accuracy: {detection_metrics['fault_classification_accuracy']*100:.2f}% "
          f"({detection_metrics['classification_correct']}/{detection_metrics['classification_total']} correct)")
    
    print_confusion_matrix(confusion, list(all_classes))
    
    # UPGRADE: Sub-type confusion matrix
    print("\n" + "=" * 70)
    print("Sub-Type Confusion Matrix")
    print("=" * 70)
    print("(Shows fine-grained bearing sub-types vs Person C predictions)\n")
    subtype_classes = set()
    for row in subtype_confusion.values():
        subtype_classes.update(row.keys())
    subtype_classes.update(subtype_confusion.keys())
    print_confusion_matrix(subtype_confusion, list(subtype_classes))
    
    print_classification_report(metrics)
    
    # UPGRADE: Macro-F1
    print(f"\n📊 Macro-F1 Score: {macro_f1:.3f}")
    
    # UPGRADE: Calibration
    print("\n" + "=" * 70)
    print("Confidence Calibration (ECE)")
    print("=" * 70)
    print(f"\nExpected Calibration Error: {calibration['expected_calibration_error']:.4f}")
    print(f"Number of bins: {calibration['n_bins']}")
    print(f"\nReliability Curve:")
    for bin_data in calibration['reliability_curve']:
        bin_range = bin_data['bin_range']
        print(f"  [{bin_range[0]:.2f}, {bin_range[1]:.2f}]: "
              f"conf={bin_data['avg_confidence']:.3f}, "
              f"acc={bin_data['avg_accuracy']:.3f}, "
              f"gap={bin_data['gap']:+.3f}, "
              f"n={bin_data['count']}")
    
    print_confidence_analysis(conf_stats)
    print_error_analysis(errors_enhanced)
    
    # Save reports
    print(f"\n3. Saving evaluation reports...")
    save_evaluation_report(results_dir, accuracy, confusion, metrics, conf_stats, errors_enhanced)
    save_error_analysis_json(results_dir, errors_enhanced)
    save_calibration_json(results_dir, calibration)
    
    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Complete (UPGRADED)!")
    print("=" * 70)
    
    print(f"\n✓ Person C achieved {accuracy*100:.2f}% accuracy on CWRU bearing dataset")
    print(f"✓ Fault detection: {detection_metrics['fault_detection_accuracy']*100:.2f}%")
    print(f"✓ Fault classification: {detection_metrics['fault_classification_accuracy']*100:.2f}%")
    print(f"✓ Macro-F1: {macro_f1:.3f}")
    print(f"✓ ECE: {calibration['expected_calibration_error']:.4f}")
    print(f"✓ Results saved to {results_dir}")
    
    # Key insights
    print(f"\n📈 Key Insights:")
    
    # Best performing class
    best_class = max(metrics.items(), key=lambda x: x[1]["f1"])
    print(f"  • Best performing class: {best_class[0]} (F1={best_class[1]['f1']:.3f})")
    
    # Worst performing class
    worst_class = min(metrics.items(), key=lambda x: x[1]["f1"])
    print(f"  • Worst performing class: {worst_class[0]} (F1={worst_class[1]['f1']:.3f})")
    
    # Confidence insight
    if "correct" in conf_stats and "incorrect" in conf_stats:
        conf_diff = conf_stats["correct"]["mean"] - conf_stats["incorrect"]["mean"]
        print(f"  • Confidence calibration: Correct predictions have {conf_diff:.3f} higher confidence on average")
    
    # Calibration insight
    if calibration['expected_calibration_error'] < 0.10:
        print(f"  • Well-calibrated: ECE < 0.10 indicates good confidence alignment")
    else:
        print(f"  • Calibration needs improvement: ECE = {calibration['expected_calibration_error']:.4f}")
    
    # Error analysis insight
    print(f"  • Total errors: {len(errors_enhanced)} ({len(errors_enhanced)/len(predictions)*100:.1f}%)")
    print(f"  • Detailed error analysis in: error_analysis.json")
    
    print(f"\n📁 Full results in: {results_dir}")
    print(f"   - evaluation_report.json")
    print(f"   - error_analysis.json")
    print(f"   - calibration.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
