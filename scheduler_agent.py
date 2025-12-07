"""
Scheduler Agent for MIRA Wave Person C

Translates diagnostic results into concrete repair recommendations
with urgency assessment, downtime estimation, and risk evaluation.

Tools (3):
1. compute_repair_urgency()
2. select_workshop_type()
3. generate_repair_plan_json()
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


class SchedulerAgent:
    """
    Agent responsible for generating actionable repair plans with
    urgency levels, workshop recommendations, and risk assessment.
    """
    
    def __init__(self):
        """Initialize Scheduler Agent with repair knowledge base."""
        
        # Define repair actions for each fault type
        self.repair_database = {
            "Loose Mount": {
                "primary_action": "Tighten engine/transmission mount bolts to specification",
                "detailed_steps": [
                    "Lift vehicle and secure on jack stands",
                    "Inspect mount rubber isolators for degradation",
                    "Retorque mounting bolts to manufacturer specification (75-95 Nm typical)",
                    "Replace rubber isolators if cracking or compression visible",
                    "Verify no contact between powertrain and chassis"
                ],
                "parts_required": [
                    "Engine mount rubber isolators (conditional)",
                    "Transmission mount bushings (conditional)",
                    "Lock washers"
                ],
                "tools_required": [
                    "Torque wrench (50-120 Nm range)",
                    "Jack and stands",
                    "Socket set",
                    "Inspection mirror"
                ],
                "base_downtime_hours": 2.5,
                "complexity": "medium",
                "workshop_type": "Engine & powertrain specialist",
                "urgency_factors": {
                    "rms_vibration_high": {"threshold": 4.0, "urgency_boost": 1},
                    "speed_dependency_strong": {"urgency_boost": 1}
                }
            },
            "Bearing Wear": {
                "primary_action": "Replace worn wheel or hub bearing assembly",
                "detailed_steps": [
                    "Lift vehicle and remove wheel",
                    "Remove brake caliper and rotor",
                    "Extract hub bearing assembly using press or puller",
                    "Clean bearing seat and inspect for damage",
                    "Press in new bearing with proper alignment",
                    "Reassemble brake components",
                    "Torque wheel bearing nut to specification (200-300 Nm typical)",
                    "Verify bearing preload and rotation smoothness"
                ],
                "parts_required": [
                    "Wheel hub bearing assembly (OEM)",
                    "Bearing retaining circlip",
                    "Hub nut",
                    "Brake cleaner"
                ],
                "tools_required": [
                    "Hydraulic press or bearing puller",
                    "Torque wrench (high range)",
                    "Socket set",
                    "Bearing installer"
                ],
                "base_downtime_hours": 3.0,
                "complexity": "high",
                "workshop_type": "Chassis & suspension specialist",
                "urgency_factors": {
                    "bearing_energy_high": {"threshold": 0.80, "urgency_boost": 2},
                    "spectral_entropy_high": {"threshold": 0.90, "urgency_boost": 1}
                }
            },
            "Imbalance": {
                "primary_action": "Perform dynamic wheel balancing",
                "detailed_steps": [
                    "Remove all wheels",
                    "Clean wheels and inspect for damage or debris",
                    "Mount on dynamic balancing machine",
                    "Measure imbalance magnitude and location",
                    "Apply balance weights to correction planes",
                    "Verify balance within 5 grams per wheel",
                    "Inspect driveshaft for missing balance weights",
                    "Reinstall wheels with proper torque sequence"
                ],
                "parts_required": [
                    "Wheel balance weights (clip-on or adhesive)",
                    "Driveshaft balance weights (if missing)"
                ],
                "tools_required": [
                    "Dynamic wheel balancer",
                    "Torque wrench",
                    "Weight hammer",
                    "Cleaning supplies"
                ],
                "base_downtime_hours": 1.5,
                "complexity": "low",
                "workshop_type": "General automotive service",
                "urgency_factors": {
                    "rms_vibration_high": {"threshold": 3.5, "urgency_boost": 0},
                    "dominant_freq_rotational": {"urgency_boost": 0}
                }
            },
            "Misalignment": {
                "primary_action": "Realign drivetrain or suspension geometry",
                "detailed_steps": [
                    "Perform four-wheel alignment measurement",
                    "Inspect CV joint boots for damage",
                    "Check engine and transmission mount alignment",
                    "Adjust suspension camber, caster, and toe to specification",
                    "Verify driveshaft angle within 1.5 degrees",
                    "Inspect universal joints for wear",
                    "Test drive to verify vibration elimination"
                ],
                "parts_required": [
                    "CV joint boot kit (if damaged)",
                    "Alignment shims (conditional)",
                    "Universal joint (conditional)"
                ],
                "tools_required": [
                    "Four-wheel alignment rack",
                    "Angle gauge",
                    "Torque wrench",
                    "CV joint tools"
                ],
                "base_downtime_hours": 4.0,
                "complexity": "high",
                "workshop_type": "Alignment & driveline specialist",
                "urgency_factors": {
                    "rms_vibration_high": {"threshold": 4.0, "urgency_boost": 1},
                    "harmonic_presence": {"urgency_boost": 1}
                }
            }
        }
        
        # Risk descriptions for ignored faults
        self.risk_descriptions = {
            "Loose Mount": {
                "immediate": "Severe cabin vibration and noise",
                "short_term": "Accelerated wear of surrounding components",
                "long_term": "Chassis structural fatigue, powertrain misalignment, secondary damage to exhaust system and cooling hoses",
                "safety_critical": False
            },
            "Bearing Wear": {
                "immediate": "Increased noise and vibration",
                "short_term": "Rapid bearing degradation",
                "long_term": "Catastrophic bearing seizure, wheel detachment risk, potential loss of vehicle control, ABS system failure",
                "safety_critical": True
            },
            "Imbalance": {
                "immediate": "Steering wheel oscillation and discomfort",
                "short_term": "Accelerated tire wear",
                "long_term": "Premature suspension bushing failure, steering component wear, reduced tire life by 30-50%",
                "safety_critical": False
            },
            "Misalignment": {
                "immediate": "Directional vibration and handling changes",
                "short_term": "Uneven tire wear and increased fuel consumption",
                "long_term": "CV joint failure, differential bearing damage, driveshaft universal joint failure, potential drivetrain disconnect",
                "safety_critical": True
            }
        }
    
    def compute_repair_urgency(
        self,
        current_run: DiagnosticRun,
        root_cause: str,
        confidence: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Tool 1: Compute repair urgency level based on fault severity and features.
        
        Args:
            current_run: Current diagnostic run
            root_cause: Identified root cause
            confidence: Confidence in root cause
            
        Returns:
            Tuple of (urgency_level, urgency_details)
        """
        repair_info = self.repair_database.get(root_cause)
        if not repair_info:
            return "Medium", {"reason": "Unknown fault type"}
        
        features = current_run.features
        urgency_score = 0
        urgency_reasons = []
        
        # Base urgency from fault type
        risk_info = self.risk_descriptions.get(root_cause, {})
        if risk_info.get("safety_critical", False):
            urgency_score += 2
            urgency_reasons.append("Safety-critical component affected")
        
        # Evaluate urgency factors specific to fault type
        urgency_factors = repair_info.get("urgency_factors", {})
        
        # RMS vibration check
        if "rms_vibration_high" in urgency_factors:
            threshold = urgency_factors["rms_vibration_high"]["threshold"]
            if features.rms_vibration >= threshold:
                boost = urgency_factors["rms_vibration_high"]["urgency_boost"]
                urgency_score += boost
                urgency_reasons.append(f"High vibration amplitude ({features.rms_vibration:.2f} > {threshold})")
        
        # Bearing energy check
        if "bearing_energy_high" in urgency_factors:
            threshold = urgency_factors["bearing_energy_high"]["threshold"]
            if features.bearing_energy_band >= threshold:
                boost = urgency_factors["bearing_energy_high"]["urgency_boost"]
                urgency_score += boost
                urgency_reasons.append(f"Advanced bearing degradation ({features.bearing_energy_band:.2f})")
        
        # Spectral entropy check
        if "spectral_entropy_high" in urgency_factors:
            threshold = urgency_factors["spectral_entropy_high"]["threshold"]
            if features.spectral_entropy >= threshold:
                boost = urgency_factors["spectral_entropy_high"]["urgency_boost"]
                urgency_score += boost
                urgency_reasons.append(f"High signal complexity ({features.spectral_entropy:.2f})")
        
        # Speed dependency check
        if "speed_dependency_strong" in urgency_factors:
            if features.speed_dependency == "strong":
                boost = urgency_factors["speed_dependency_strong"]["urgency_boost"]
                urgency_score += boost
                urgency_reasons.append("Strong speed-dependent behavior")
        
        # Confidence adjustment
        if confidence < 0.70:
            urgency_score -= 1
            urgency_reasons.append("Reduced urgency due to diagnostic uncertainty")
        
        # Map score to urgency level
        if urgency_score >= 3:
            urgency_level = "High"
        elif urgency_score >= 1:
            urgency_level = "Medium"
        else:
            urgency_level = "Low"
        
        urgency_details = {
            "urgency_score": urgency_score,
            "reasons": urgency_reasons,
            "safety_critical": risk_info.get("safety_critical", False)
        }
        
        return urgency_level, urgency_details
    
    def select_workshop_type(
        self,
        root_cause: str,
        complexity: str = None
    ) -> Dict[str, Any]:
        """
        Tool 2: Select appropriate workshop type and required capabilities.
        
        Args:
            root_cause: Identified root cause
            complexity: Optional complexity override
            
        Returns:
            Workshop selection dictionary with capabilities and certifications
        """
        repair_info = self.repair_database.get(root_cause)
        if not repair_info:
            return {
                "workshop_type": "General automotive service",
                "required_capabilities": ["Basic diagnostic tools"],
                "certifications": [],
                "equipment": []
            }
        
        workshop_type = repair_info["workshop_type"]
        complexity = complexity or repair_info["complexity"]
        tools = repair_info["tools_required"]
        
        # Define capability requirements by workshop type
        workshop_capabilities = {
            "Engine & powertrain specialist": {
                "required_capabilities": [
                    "Engine diagnostics",
                    "Powertrain mount replacement",
                    "Torque specification expertise"
                ],
                "certifications": ["ASE Engine Repair certification"],
                "equipment": tools + ["Engine lift equipment", "Diagnostic scanner"]
            },
            "Chassis & suspension specialist": {
                "required_capabilities": [
                    "Bearing replacement expertise",
                    "Hydraulic press operation",
                    "Suspension diagnostics"
                ],
                "certifications": ["ASE Suspension & Steering certification"],
                "equipment": tools + ["Hydraulic press", "Bearing installer kit"]
            },
            "General automotive service": {
                "required_capabilities": [
                    "Wheel balancing",
                    "Basic maintenance"
                ],
                "certifications": ["General automotive certification"],
                "equipment": tools
            },
            "Alignment & driveline specialist": {
                "required_capabilities": [
                    "Four-wheel alignment",
                    "Driveline diagnostics",
                    "CV joint service"
                ],
                "certifications": ["ASE Alignment certification", "Driveline specialist"],
                "equipment": tools + ["Alignment rack", "Angle measurement tools"]
            }
        }
        
        workshop_info = workshop_capabilities.get(workshop_type, {
            "required_capabilities": ["General repair"],
            "certifications": [],
            "equipment": tools
        })
        
        return {
            "workshop_type": workshop_type,
            "complexity": complexity,
            **workshop_info
        }
    
    def generate_repair_plan_json(
        self,
        current_run: DiagnosticRun,
        root_cause: str,
        confidence: float,
        experiment_performed: bool,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Tool 3: Generate complete repair plan in repair.json format.
        
        Args:
            current_run: Current diagnostic run
            root_cause: Identified root cause
            confidence: Confidence in diagnosis
            experiment_performed: Whether active experiment was performed
            output_dir: Directory to save output
            
        Returns:
            Dictionary in repair.json format
        """
        repair_info = self.repair_database.get(root_cause)
        if not repair_info:
            # Fallback for unknown fault
            repair_json = {
                "run_id": current_run.run_id,
                "recommended_repair": "Comprehensive diagnostic inspection required",
                "urgency": "Medium",
                "estimated_downtime_hours": 2.0,
                "risk_if_ignored": "Uncertain - requires further investigation",
                "suggested_workshop_type": "General automotive service"
            }
        else:
            # Compute urgency
            urgency_level, urgency_details = self.compute_repair_urgency(
                current_run,
                root_cause,
                confidence
            )
            
            # Select workshop
            workshop_info = self.select_workshop_type(root_cause)
            
            # Estimate downtime
            base_downtime = repair_info["base_downtime_hours"]
            
            # Adjust for complexity and parts availability
            if urgency_details.get("safety_critical"):
                # Expedite for safety issues
                downtime_multiplier = 0.9
            elif confidence < 0.75:
                # Extra diagnostic time for uncertainty
                downtime_multiplier = 1.3
            else:
                downtime_multiplier = 1.0
            
            estimated_downtime = base_downtime * downtime_multiplier
            
            # Build risk description
            risk_info = self.risk_descriptions.get(root_cause, {})
            risk_parts = []
            
            if urgency_level == "High":
                risk_parts.append(risk_info.get("immediate", "Immediate operational issues"))
                risk_parts.append(risk_info.get("long_term", "Potential system failure"))
            elif urgency_level == "Medium":
                risk_parts.append(risk_info.get("short_term", "Accelerated component wear"))
                risk_parts.append(risk_info.get("long_term", "Extended damage to related systems"))
            else:
                risk_parts.append(risk_info.get("immediate", "Minor discomfort"))
                risk_parts.append(risk_info.get("short_term", "Gradual component degradation"))
            
            risk_description = ". ".join(risk_parts) + "."
            
            repair_json = {
                "run_id": current_run.run_id,
                "recommended_repair": repair_info["primary_action"],
                "detailed_steps": repair_info["detailed_steps"],
                "urgency": urgency_level,
                "urgency_details": {
                    "score": urgency_details["urgency_score"],
                    "reasons": urgency_details["reasons"],
                    "safety_critical": urgency_details["safety_critical"]
                },
                "estimated_downtime_hours": round(estimated_downtime, 1),
                "parts_required": repair_info["parts_required"],
                "tools_required": repair_info["tools_required"],
                "complexity": repair_info["complexity"],
                "risk_if_ignored": risk_description,
                "suggested_workshop_type": workshop_info["workshop_type"],
                "workshop_requirements": {
                    "capabilities": workshop_info["required_capabilities"],
                    "certifications": workshop_info["certifications"],
                    "equipment": workshop_info["equipment"]
                },
                "cost_estimate_range": self._estimate_cost_range(root_cause, repair_info),
                "experiment_informed": experiment_performed,
                "confidence_in_diagnosis": round(confidence, 3)
            }
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"repair_{current_run.run_id}.json"
        save_json(repair_json, output_path)
        
        return repair_json
    
    def _estimate_cost_range(
        self,
        root_cause: str,
        repair_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Internal helper to estimate repair cost range.
        
        Args:
            root_cause: Fault type
            repair_info: Repair information dictionary
            
        Returns:
            Dictionary with min and max cost estimates (USD)
        """
        # Base cost estimates by fault type (parts + labor)
        cost_ranges = {
            "Loose Mount": {"min": 150, "max": 400},
            "Bearing Wear": {"min": 250, "max": 600},
            "Imbalance": {"min": 50, "max": 150},
            "Misalignment": {"min": 300, "max": 800}
        }
        
        base_range = cost_ranges.get(root_cause, {"min": 100, "max": 500})
        
        # Adjust for complexity
        complexity = repair_info.get("complexity", "medium")
        if complexity == "high":
            multiplier = 1.3
        elif complexity == "low":
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        return {
            "min_usd": round(base_range["min"] * multiplier),
            "max_usd": round(base_range["max"] * multiplier),
            "currency": "USD",
            "includes": "Parts and labor estimate"
        }
    
    def process_run(
        self,
        current_run: DiagnosticRun,
        root_cause: str,
        confidence: float,
        experiment_required: bool
    ) -> Dict[str, Any]:
        """
        Main processing method that orchestrates all scheduling tools.
        
        Args:
            current_run: Current diagnostic run
            root_cause: Root cause from causal inference
            confidence: Confidence in diagnosis
            experiment_required: Whether experiment was needed/performed
            
        Returns:
            Comprehensive repair scheduling results
        """
        # Compute urgency
        urgency_level, urgency_details = self.compute_repair_urgency(
            current_run,
            root_cause,
            confidence
        )
        
        # Select workshop
        workshop_info = self.select_workshop_type(root_cause)
        
        # Get repair details
        repair_info = self.repair_database.get(root_cause, {})
        
        return {
            "run_id": current_run.run_id,
            "root_cause": root_cause,
            "urgency_level": urgency_level,
            "urgency_details": urgency_details,
            "workshop_info": workshop_info,
            "repair_action": repair_info.get("primary_action", "Diagnostic inspection required"),
            "estimated_downtime": repair_info.get("base_downtime_hours", 2.0),
            "complexity": repair_info.get("complexity", "medium")
        }
