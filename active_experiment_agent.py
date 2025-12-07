"""
Active Experiment Agent for MIRA Wave Person C

Designs uncertainty-reduction experiments when diagnostic confidence
is below threshold, predicting information gain and generating test protocols.

Tools (5):
1. check_uncertainty_threshold()
2. design_new_speed_profile()
3. design_load_change()
4. predict_information_gain()
5. generate_experiment_instruction()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from .data_models import DiagnosticRun
    from .utils import save_json
except ImportError:
    from data_models import DiagnosticRun
    from utils import save_json


class ActiveExperimentAgent:
    """
    Agent responsible for designing active experiments to reduce
    diagnostic uncertainty when confidence thresholds are not met.
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize Active Experiment Agent.
        
        Args:
            confidence_threshold: Minimum confidence to skip experiments (default 0.75)
        """
        self.confidence_threshold = confidence_threshold
        
        # Define experiment types and their expected effects per fault
        self.experiment_signatures = {
            "increase_speed": {
                "Loose Mount": {
                    "description": "Increase vehicle speed from 40 to 80 km/h",
                    "expected_behavior": "Nonlinear vibration amplitude increase due to resonance amplification at natural frequencies",
                    "key_indicators": ["rms_vibration", "dominant_frequency"],
                    "expected_change": {"rms_vibration": 1.8, "dominant_frequency": 1.3},
                    "information_gain": 0.25
                },
                "Bearing Wear": {
                    "description": "Increase vehicle speed from 40 to 80 km/h",
                    "expected_behavior": "Linear increase in bearing frequency components with slight amplitude growth",
                    "key_indicators": ["bearing_energy_band", "dominant_frequency"],
                    "expected_change": {"bearing_energy_band": 1.15, "dominant_frequency": 2.0},
                    "information_gain": 0.18
                },
                "Imbalance": {
                    "description": "Increase vehicle speed from 40 to 80 km/h",
                    "expected_behavior": "Quadratic vibration amplitude growth proportional to rotational speed squared",
                    "key_indicators": ["rms_vibration", "dominant_frequency"],
                    "expected_change": {"rms_vibration": 2.2, "dominant_frequency": 2.0},
                    "information_gain": 0.28
                },
                "Misalignment": {
                    "description": "Increase vehicle speed from 40 to 80 km/h",
                    "expected_behavior": "Proportional increase in harmonic components with preserved frequency ratios",
                    "key_indicators": ["rms_vibration", "dominant_frequency"],
                    "expected_change": {"rms_vibration": 1.6, "dominant_frequency": 2.0},
                    "information_gain": 0.20
                }
            },
            "increase_load": {
                "Loose Mount": {
                    "description": "Add 200 kg load to vehicle cargo area",
                    "expected_behavior": "Significant vibration amplitude increase and frequency shift due to altered suspension dynamics",
                    "key_indicators": ["rms_vibration", "spectral_entropy"],
                    "expected_change": {"rms_vibration": 1.5, "spectral_entropy": 1.12},
                    "information_gain": 0.22
                },
                "Bearing Wear": {
                    "description": "Add 200 kg load to vehicle cargo area",
                    "expected_behavior": "Minimal change in bearing frequencies, slight increase in bearing band energy",
                    "key_indicators": ["bearing_energy_band", "rms_vibration"],
                    "expected_change": {"bearing_energy_band": 1.08, "rms_vibration": 1.1},
                    "information_gain": 0.15
                },
                "Imbalance": {
                    "description": "Add 200 kg load to vehicle cargo area",
                    "expected_behavior": "Minor vibration changes, imbalance signature remains dominant",
                    "key_indicators": ["rms_vibration", "dominant_frequency"],
                    "expected_change": {"rms_vibration": 1.15, "dominant_frequency": 1.05},
                    "information_gain": 0.12
                },
                "Misalignment": {
                    "description": "Add 200 kg load to vehicle cargo area",
                    "expected_behavior": "Moderate vibration increase with altered load paths affecting misalignment stress",
                    "key_indicators": ["rms_vibration", "spectral_entropy"],
                    "expected_change": {"rms_vibration": 1.35, "spectral_entropy": 1.08},
                    "information_gain": 0.19
                }
            },
            "apply_braking": {
                "Loose Mount": {
                    "description": "Apply moderate braking torque (0.3g deceleration)",
                    "expected_behavior": "Asymmetric torque transfer amplifies low-frequency structural vibration by 30-50%",
                    "key_indicators": ["rms_vibration", "speed_dependency"],
                    "expected_change": {"rms_vibration": 1.45, "spectral_entropy": 1.15},
                    "information_gain": 0.26
                },
                "Bearing Wear": {
                    "description": "Apply moderate braking torque (0.3g deceleration)",
                    "expected_behavior": "Minimal impact on bearing signature, slight load redistribution",
                    "key_indicators": ["bearing_energy_band", "rms_vibration"],
                    "expected_change": {"bearing_energy_band": 1.05, "rms_vibration": 1.05},
                    "information_gain": 0.10
                },
                "Imbalance": {
                    "description": "Apply moderate braking torque (0.3g deceleration)",
                    "expected_behavior": "Imbalance remains unaffected by braking load, rotational signature preserved",
                    "key_indicators": ["rms_vibration", "dominant_frequency"],
                    "expected_change": {"rms_vibration": 1.02, "dominant_frequency": 1.0},
                    "information_gain": 0.24
                },
                "Misalignment": {
                    "description": "Apply moderate braking torque (0.3g deceleration)",
                    "expected_behavior": "Torque redistribution increases axial stress and misalignment vibration",
                    "key_indicators": ["rms_vibration", "spectral_entropy"],
                    "expected_change": {"rms_vibration": 1.30, "spectral_entropy": 1.10},
                    "information_gain": 0.21
                }
            },
            "road_roughness": {
                "Loose Mount": {
                    "description": "Drive over standardized rough road section (ISO 8608 Class C)",
                    "expected_behavior": "Excitation of structural modes amplifies loose mount vibration across broad frequency range",
                    "key_indicators": ["rms_vibration", "spectral_entropy"],
                    "expected_change": {"rms_vibration": 2.0, "spectral_entropy": 1.25},
                    "information_gain": 0.30
                },
                "Bearing Wear": {
                    "description": "Drive over standardized rough road section (ISO 8608 Class C)",
                    "expected_behavior": "Impact loading may temporarily elevate bearing noise but frequency signature remains",
                    "key_indicators": ["bearing_energy_band", "audio_anomaly_score"],
                    "expected_change": {"bearing_energy_band": 1.12, "audio_anomaly_score": 1.20},
                    "information_gain": 0.16
                },
                "Imbalance": {
                    "description": "Drive over standardized rough road section (ISO 8608 Class C)",
                    "expected_behavior": "Road excitation adds noise but imbalance frequency remains distinct",
                    "key_indicators": ["spectral_entropy", "rms_vibration"],
                    "expected_change": {"spectral_entropy": 1.18, "rms_vibration": 1.4},
                    "information_gain": 0.14
                },
                "Misalignment": {
                    "description": "Drive over standardized rough road section (ISO 8608 Class C)",
                    "expected_behavior": "Complex loading conditions stress misaligned components, increasing vibration",
                    "key_indicators": ["rms_vibration", "spectral_entropy"],
                    "expected_change": {"rms_vibration": 1.65, "spectral_entropy": 1.20},
                    "information_gain": 0.22
                }
            }
        }
    
    def check_uncertainty_threshold(
        self,
        confidence: float,
        cause_probabilities: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Tool 1: Check if diagnostic uncertainty requires an active experiment.
        
        Args:
            confidence: Confidence in root cause prediction
            cause_probabilities: Probability distribution over causes
            
        Returns:
            Tuple of (experiment_required, uncertainty_level)
        """
        experiment_required = confidence < self.confidence_threshold
        
        # Compute entropy-based uncertainty
        probs = list(cause_probabilities.values())
        probs = [p for p in probs if p > 0]
        
        if probs:
            entropy = -sum(p * np.log2(p) for p in probs)
            max_entropy = np.log2(len(probs))
            normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_uncertainty = 1.0
        
        return experiment_required, float(normalized_uncertainty)
    
    def design_new_speed_profile(
        self,
        current_run: DiagnosticRun,
        hypothesized_cause: str
    ) -> Dict[str, Any]:
        """
        Tool 2: Design speed variation experiment protocol.
        
        Args:
            current_run: Current diagnostic run
            hypothesized_cause: Most probable cause from causal inference
            
        Returns:
            Speed experiment design dictionary
        """
        signature = self.experiment_signatures["increase_speed"][hypothesized_cause]
        
        experiment_design = {
            "experiment_type": "increase_speed",
            "protocol": {
                "initial_speed_kmh": 40,
                "final_speed_kmh": 80,
                "acceleration_rate": "moderate (2 m/s²)",
                "test_duration_seconds": 30,
                "road_condition": "smooth asphalt",
                "gear": "automatic or 4th gear manual"
            },
            "measurement_requirements": {
                "sample_rate_hz": 1000,
                "sensors": ["accelerometer_chassis", "accelerometer_engine", "microphone_cabin"],
                "speed_intervals": [40, 50, 60, 70, 80]
            },
            "safety_constraints": {
                "max_speed_kmh": 80,
                "required_road_length_m": 500,
                "weather": "dry conditions only"
            }
        }
        
        return experiment_design
    
    def design_load_change(
        self,
        current_run: DiagnosticRun,
        hypothesized_cause: str
    ) -> Dict[str, Any]:
        """
        Tool 3: Design load variation experiment protocol.
        
        Args:
            current_run: Current diagnostic run
            hypothesized_cause: Most probable cause from causal inference
            
        Returns:
            Load experiment design dictionary
        """
        signature = self.experiment_signatures["increase_load"][hypothesized_cause]
        
        experiment_design = {
            "experiment_type": "increase_load",
            "protocol": {
                "baseline_load_kg": 0,
                "test_load_kg": 200,
                "load_position": "center of cargo area",
                "stabilization_time_seconds": 60
            },
            "measurement_requirements": {
                "sample_rate_hz": 1000,
                "test_speed_kmh": 50,
                "test_duration_seconds": 120,
                "sensors": ["accelerometer_chassis", "suspension_displacement"]
            },
            "safety_constraints": {
                "max_load_kg": 200,
                "vehicle_load_rating_check": "required",
                "secure_load": "properly fastened"
            }
        }
        
        return experiment_design
    
    def predict_information_gain(
        self,
        current_run: DiagnosticRun,
        hypothesized_cause: str,
        alternate_causes: List[str],
        experiment_type: str
    ) -> Dict[str, float]:
        """
        Tool 4: Predict information gain from each experiment type for cause disambiguation.
        
        Args:
            current_run: Current diagnostic run
            hypothesized_cause: Most probable cause
            alternate_causes: List of alternate causes to discriminate
            experiment_type: Type of experiment to evaluate
            
        Returns:
            Dictionary mapping causes to predicted information gain
        """
        if experiment_type not in self.experiment_signatures:
            return {cause: 0.0 for cause in [hypothesized_cause] + alternate_causes}
        
        experiment_data = self.experiment_signatures[experiment_type]
        
        # Calculate discriminative power
        hyp_gain = experiment_data[hypothesized_cause]["information_gain"]
        
        gains = {hypothesized_cause: hyp_gain}
        
        for alt_cause in alternate_causes:
            if alt_cause in experiment_data:
                alt_gain = experiment_data[alt_cause]["information_gain"]
                # Information gain is higher when experiment differentiates causes better
                discriminative_power = abs(hyp_gain - alt_gain)
                gains[alt_cause] = discriminative_power
        
        return gains
    
    def generate_experiment_instruction(
        self,
        current_run: DiagnosticRun,
        hypothesized_cause: str,
        alternate_causes: List[str],
        cause_probabilities: Dict[str, float],
        uncertainty_level: float,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Tool 5: Generate complete experiment instruction in experiment.json format.
        
        Args:
            current_run: Current diagnostic run
            hypothesized_cause: Most probable cause
            alternate_causes: List of alternate probable causes
            cause_probabilities: Probability distribution over causes
            uncertainty_level: Computed uncertainty level
            output_dir: Directory to save output
            
        Returns:
            Dictionary in experiment.json format
        """
        # Determine if experiment is required
        experiment_required, _ = self.check_uncertainty_threshold(
            cause_probabilities.get(hypothesized_cause, 0),
            cause_probabilities
        )
        
        if not experiment_required:
            experiment_json = {
                "run_id": current_run.run_id,
                "experiment_required": False,
                "reason": f"Confidence {cause_probabilities.get(hypothesized_cause, 0):.2f} exceeds threshold {self.confidence_threshold}",
                "recommended_experiment": None,
                "instruction": "No experiment needed. Proceed directly to repair recommendation.",
                "expected_signal_change": None,
                "uncertainty_reduction_estimate": 0.0
            }
        else:
            # Select best experiment type
            experiment_gains = {}
            for exp_type in self.experiment_signatures.keys():
                gains = self.predict_information_gain(
                    current_run,
                    hypothesized_cause,
                    alternate_causes[:2],  # Top 2 alternates
                    exp_type
                )
                # Average gain across all causes
                avg_gain = np.mean(list(gains.values()))
                experiment_gains[exp_type] = avg_gain
            
            # Select experiment with highest gain
            best_experiment = max(experiment_gains, key=experiment_gains.get)
            
            # Get experiment signature
            exp_signature = self.experiment_signatures[best_experiment][hypothesized_cause]
            
            # Design experiment protocol
            if best_experiment == "increase_speed":
                protocol = self.design_new_speed_profile(current_run, hypothesized_cause)
            elif best_experiment == "increase_load":
                protocol = self.design_load_change(current_run, hypothesized_cause)
            elif best_experiment == "apply_braking":
                protocol = {
                    "experiment_type": "apply_braking",
                    "protocol": {
                        "initial_speed_kmh": 60,
                        "deceleration_g": 0.3,
                        "brake_duration_seconds": 3
                    }
                }
            else:  # road_roughness
                protocol = {
                    "experiment_type": "road_roughness",
                    "protocol": {
                        "road_class": "ISO 8608 Class C",
                        "test_speed_kmh": 30,
                        "section_length_m": 100
                    }
                }
            
            # Format instruction
            instruction = f"{exp_signature['description']}. {exp_signature['expected_behavior']}"
            
            # Estimate uncertainty reduction
            base_info_gain = exp_signature["information_gain"]
            uncertainty_reduction = min(uncertainty_level * base_info_gain * 1.5, 0.35)
            
            experiment_json = {
                "run_id": current_run.run_id,
                "experiment_required": True,
                "recommended_experiment": best_experiment,
                "instruction": instruction,
                "expected_signal_change": exp_signature["expected_behavior"],
                "uncertainty_reduction_estimate": round(uncertainty_reduction, 3),
                "protocol": protocol.get("protocol", {}),
                "key_indicators": exp_signature["key_indicators"],
                "expected_feature_changes": exp_signature["expected_change"]
            }
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"experiment_{current_run.run_id}.json"
        save_json(experiment_json, output_path)
        
        return experiment_json
    
    def process_run(
        self,
        current_run: DiagnosticRun,
        root_cause: str,
        cause_probabilities: Dict[str, float],
        ranked_causes: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Main processing method that orchestrates all experiment design tools.
        
        Args:
            current_run: Current diagnostic run
            root_cause: Root cause from causal inference
            cause_probabilities: Probability distribution from causal inference
            ranked_causes: Ranked list of (cause, probability) tuples
            
        Returns:
            Comprehensive experiment design results
        """
        confidence = cause_probabilities.get(root_cause, 0)
        
        # Check uncertainty
        experiment_required, uncertainty_level = self.check_uncertainty_threshold(
            confidence,
            cause_probabilities
        )
        
        # Get alternate causes
        alternate_causes = [cause for cause, _ in ranked_causes[1:4]]
        
        # Predict information gain for each experiment type
        experiment_evaluations = {}
        for exp_type in self.experiment_signatures.keys():
            gains = self.predict_information_gain(
                current_run,
                root_cause,
                alternate_causes,
                exp_type
            )
            experiment_evaluations[exp_type] = gains
        
        return {
            "run_id": current_run.run_id,
            "experiment_required": experiment_required,
            "uncertainty_level": uncertainty_level,
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "experiment_evaluations": experiment_evaluations,
            "hypothesized_cause": root_cause,
            "alternate_causes": alternate_causes
        }
