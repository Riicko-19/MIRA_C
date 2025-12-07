"""
Pipeline runner for MIRA Wave Person C diagnostic system.

Full implementation: Fleet Matching, Causal Inference, Active Experiments, Repair Scheduling, and Explanation.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings

try:
    from .data_models import DiagnosticRun, FeatureVector, FaultLocation
    from .fleet_matching_agent import FleetMatchingAgent
    from .causal_inference_agent import CausalInferenceAgent
    from .active_experiment_agent import ActiveExperimentAgent
    from .scheduler_agent import SchedulerAgent
    from .explanation_agent import ExplanationAgent
    from .utils import validate_diagnostic_input
except ImportError:
    from data_models import DiagnosticRun, FeatureVector, FaultLocation
    from fleet_matching_agent import FleetMatchingAgent
    from causal_inference_agent import CausalInferenceAgent
    from active_experiment_agent import ActiveExperimentAgent
    from scheduler_agent import SchedulerAgent
    from explanation_agent import ExplanationAgent
    from utils import validate_diagnostic_input


# Legacy wrapper for backward compatibility
def validate_input_data(run_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Legacy validation wrapper - calls central validation gate.
    
    Deprecated: Use validate_diagnostic_input() from utils.py directly.
    """
    return validate_diagnostic_input(run_data)


def load_real_fleet_database(fleet_db_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Load real fleet history from JSON file.
    
    Args:
        fleet_db_path: Path to fleet database JSON file
        
    Returns:
        List of fleet fingerprints or None if file not found
    """
    if not fleet_db_path.exists():
        return None
    
    try:
        with open(fleet_db_path, 'r') as f:
            fleet_data = json.load(f)
        
        # Convert to internal format
        processed_fleet = []
        for entry in fleet_data:
            processed_fleet.append({
                "vehicle_id": entry["vehicle_id"],
                "run_id": entry["run_id"],
                "features": FeatureVector.from_dict(entry["features"]),
                "fault_location": FaultLocation.from_dict(entry["fault_location"]),
                "known_cause": entry["known_cause"]
            })
        
        return processed_fleet
    except Exception as e:
        warnings.warn(f"Failed to load fleet database: {e}")
        return None


def run_person_c_pipeline(
    input_runs: List[Dict[str, Any]],
    output_dir: Path = None,
    fleet_database_path: Path = None,
    validate_inputs: bool = True,
    confidence_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Execute Person C multi-agent diagnostic pipeline.
    
    Complete implementation with all 5 agents:
    - Fleet Matching: Pattern similarity and clustering
    - Causal Inference: Bayesian fault attribution
    - Active Experiments: Uncertainty reduction protocols
    - Repair Scheduling: Actionable repair plans
    - Explanation: Human-readable diagnostic reports
    
    Args:
        input_runs: List of diagnostic run dictionaries
        output_dir: Optional output directory for saving results
        fleet_database_path: Optional path to real fleet database JSON
        validate_inputs: Whether to validate input data quality (recommended for real data)
        confidence_threshold: Confidence threshold for experiment requirement (default 0.75)
        
    Returns:
        Dictionary containing results for all runs with complete diagnostic chain
    """
    if output_dir is None:
        output_dir = Path("./output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs if requested
    validated_runs = []
    run_validation_info = {}  # Store validation info per run
    validation_summary = {"total": len(input_runs), "valid": 0, "invalid": 0, "warnings": 0}
    
    if validate_inputs:
        print("Validating input data...")
        for run_data in input_runs:
            is_valid, errors, warnings_list = validate_input_data(run_data)
            run_id = run_data.get('run_id', 'unknown')
            
            # Store validation info for this run
            run_validation_info[run_id] = {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings_list
            }
            
            if is_valid:
                validated_runs.append(run_data)
                validation_summary["valid"] += 1
                
                if warnings_list:
                    validation_summary["warnings"] += len(warnings_list)
                    print(f"  ⚠ {run_id}: {len(warnings_list)} warning(s)")
                    for warning in warnings_list:
                        print(f"    - {warning}")
            else:
                validation_summary["invalid"] += 1
                print(f"  ✗ {run_id}: INVALID")
                for error in errors:
                    print(f"    - {error}")
        
        print(f"Validation complete: {validation_summary['valid']}/{validation_summary['total']} runs valid\n")
        
        if validation_summary["invalid"] > 0:
            print(f"Skipping {validation_summary['invalid']} invalid run(s)\n")
    else:
        validated_runs = input_runs
        run_validation_info = {}  # No validation performed
    
    if not validated_runs:
        print("No valid runs to process. Exiting.")
        return {"validation_summary": validation_summary, "results": {}}
    
    # Initialize agents
    fleet_agent = FleetMatchingAgent(fleet_size=100)
    
    # Load real fleet database if provided
    if fleet_database_path:
        print(f"Loading real fleet database from {fleet_database_path}...")
        real_fleet = load_real_fleet_database(fleet_database_path)
        if real_fleet:
            fleet_agent.fleet_fingerprints = real_fleet
            print(f"✓ Loaded {len(real_fleet)} real fleet cases\n")
        else:
            print(f"⚠ Could not load fleet database, using synthetic fleet\n")
    
    causal_agent = CausalInferenceAgent()
    experiment_agent = ActiveExperimentAgent(confidence_threshold=confidence_threshold)
    scheduler_agent = SchedulerAgent()
    explanation_agent = ExplanationAgent()
    
    results = {}
    print(f"Processing {len(validated_runs)} diagnostic run(s)...\n")
    
    for idx, run_data in enumerate(validated_runs, 1):
        print(f"[{idx}/{len(validated_runs)}] Processing {run_data.get('run_id', 'unknown')}...")
        
        # Convert to DiagnosticRun object
        diagnostic_run = DiagnosticRun.from_dict(run_data)
        
        # Phase 1: Fleet Matching
        fleet_results = fleet_agent.process_run(diagnostic_run)
        
        # Extract fleet matches and cause probabilities
        k_neighbors = 15
        fleet_matches = fleet_agent.knn_match(diagnostic_run, k=k_neighbors)
        fleet_cause_probs = fleet_results["cause_probabilities_from_fleet"]
        
        # Phase 2: Causal Inference (with Stage-0 Normal Detection)
        causal_results = causal_agent.process_run(
            diagnostic_run,
            fleet_matches,
            fleet_cause_probs
        )
        
        # UPGRADE: Check Stage-0 decision
        stage0 = causal_results.get("stage0_normal_detection", {})
        is_normal_stage0 = stage0.get("is_normal", False)
        normal_confidence = stage0.get("normal_confidence", 0.0)
        
        # Export cause.json
        ranked_causes = causal_results["ranked_causes"]
        posteriors = causal_results["posteriors"]
        confidence_intervals = causal_results["confidence_intervals"]
        
        cause_json = causal_agent.export_causal_json(
            diagnostic_run,
            posteriors,
            ranked_causes,
            confidence_intervals,
            output_dir
        )
        
        # UPGRADE: If Stage-0 classified as Normal with high confidence, bypass experiments
        root_cause = causal_results["root_cause"]
        
        if is_normal_stage0 and normal_confidence >= 0.70:
            # Bypass Active Experiment and Scheduler for Normal cases
            print(f"  → Stage-0: NORMAL detected ({normal_confidence*100:.1f}% confidence), bypassing experiments")
            
            # CRITICAL: Override root_cause to "Normal" (Bayesian may have chosen a fault)
            root_cause = "Normal"
            
            # Phase 3: Skip Active Experiment (not needed for Normal)
            experiment_json = {
                "run_id": diagnostic_run.run_id,
                "experiment_required": False,
                "reason": f"Stage-0 classified as Normal ({normal_confidence*100:.1f}% confidence)",
                "root_cause": "Normal",
                "confidence": normal_confidence
            }
            
            # Phase 4: Minimal Scheduler (no repair needed)
            repair_json = {
                "run_id": diagnostic_run.run_id,
                "root_cause": "Normal",
                "priority": "none",
                "urgency": "none",
                "estimated_repair_time": "0 hours",
                "estimated_downtime_hours": 0,
                "repair_actions": ["No repair required - system operating normally"],
                "recommended_repair": "No repair required - system operating normally",
                "confidence": normal_confidence
            }
            
            experiment_performed = False
            confidence = normal_confidence
            
        else:
            # Standard pipeline for Faulty cases
            # Phase 3: Active Experiment Design
            experiment_results = experiment_agent.process_run(
                diagnostic_run,
                root_cause,
                posteriors,
                ranked_causes
            )
            
            # Generate experiment.json
            experiment_json = experiment_agent.generate_experiment_instruction(
                diagnostic_run,
                root_cause,
                experiment_results["alternate_causes"],
                posteriors,
                experiment_results["uncertainty_level"],
                output_dir
            )
            
            # Phase 4: Repair Scheduling
            experiment_performed = experiment_results["experiment_required"]
            confidence = posteriors.get(root_cause, 0)
            
            scheduler_results = scheduler_agent.process_run(
                diagnostic_run,
                root_cause,
                confidence,
                experiment_performed
            )
            
            # Generate repair.json
            repair_json = scheduler_agent.generate_repair_plan_json(
                diagnostic_run,
                root_cause,
                confidence,
                experiment_performed,
                output_dir
            )
        
        # Phase 5: Explanation Generation
        explanation_results = explanation_agent.process_run(
            diagnostic_run,
            cause_json,
            experiment_json,
            repair_json,
            fleet_results,
            output_format="both"  # Generate both txt and json
        )
        
        # Save text report
        if "report_text" in explanation_results:
            explanation_agent.save_explanation_txt(
                explanation_results["report_text"],
                diagnostic_run.run_id,
                output_dir
            )
        
        # Store complete results
        results[diagnostic_run.run_id] = {
            "fleet_matching": fleet_results,
            "causal_inference": causal_results,
            "cause_json": cause_json,
            "active_experiment": experiment_results,
            "experiment_json": experiment_json,
            "repair_schedule": scheduler_results,
            "repair_json": repair_json,
            "explanation": explanation_results,
            "validation": run_validation_info.get(diagnostic_run.run_id, {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }),
            "summary": {
                "run_id": diagnostic_run.run_id,
                "root_cause": root_cause,
                "confidence": confidence,
                "urgency": repair_json["urgency"],
                "experiment_required": experiment_performed,
                "estimated_downtime_hours": repair_json["estimated_downtime_hours"]
            }
        }
        
        print(f"  ✓ Completed {diagnostic_run.run_id}\n")
    
    # Add metadata to results
    results["_metadata"] = {
        "total_runs_processed": len(validated_runs),
        "validation_summary": validation_summary if validate_inputs else None,
        "fleet_source": "real_database" if fleet_database_path and load_real_fleet_database(fleet_database_path) else "synthetic",
        "confidence_threshold": confidence_threshold
    }
    
    return results
