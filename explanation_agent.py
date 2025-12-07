"""
Explanation Agent for MIRA Wave Person C

Generates human-readable explanations of diagnostic results,
tying together fleet analysis, causal reasoning, experiments, and repair plans.

Tools (5):
1. summarize_fault_location()
2. summarize_fingerprint_change()
3. summarize_causal_reasoning()
4. generate_human_readable_report()
5. save_explanation_txt()
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

try:
    from .data_models import DiagnosticRun, FaultLocation, FeatureVector
except ImportError:
    from data_models import DiagnosticRun, FaultLocation, FeatureVector


class ExplanationAgent:
    """
    Agent responsible for generating clear, human-readable explanations
    that integrate all diagnostic findings into a coherent narrative.
    """
    
    def __init__(self):
        """Initialize Explanation Agent."""
        self.vehicle_component_map = {
            (0.0, 0.3): {"x_region": "front", "y_region": "left"},
            (0.0, 0.7): {"x_region": "front", "y_region": "center-left"},
            (0.3, 0.3): {"x_region": "mid-front", "y_region": "left"},
            (0.3, 0.7): {"x_region": "mid-front", "y_region": "center"},
            (0.7, 0.3): {"x_region": "rear", "y_region": "left"},
            (0.7, 0.7): {"x_region": "rear", "y_region": "center"},
        }
    
    def summarize_fault_location(
        self,
        fault_location: FaultLocation
    ) -> str:
        """
        Tool 1: Generate human-readable fault location description.
        
        Args:
            fault_location: Fault location on 2D vehicle map
            
        Returns:
            Natural language location description
        """
        x, y = fault_location.x, fault_location.y
        
        # Determine longitudinal position
        if x < 0.33:
            x_desc = "front"
        elif x < 0.67:
            x_desc = "middle"
        else:
            x_desc = "rear"
        
        # Determine lateral position
        if y < 0.33:
            y_desc = "left side"
        elif y < 0.67:
            y_desc = "center"
        else:
            y_desc = "right side"
        
        # Map to likely components
        if x < 0.4 and y < 0.4:
            component = "front-left wheel/suspension area"
        elif x < 0.4 and y > 0.6:
            component = "front-right wheel/suspension area"
        elif x < 0.4:
            component = "engine/transmission mounting area"
        elif x > 0.6 and y < 0.4:
            component = "rear-left wheel/axle area"
        elif x > 0.6 and y > 0.6:
            component = "rear-right wheel/axle area"
        elif x > 0.6:
            component = "rear axle/differential area"
        else:
            component = "central drivetrain area"
        
        location_summary = f"Fault localized to {x_desc} section, {y_desc} ({component})"
        detailed_coords = f"Map coordinates: X={x:.2f}, Y={y:.2f}"
        
        return f"{location_summary}. {detailed_coords}"
    
    def summarize_fingerprint_change(
        self,
        features: FeatureVector,
        root_cause: str
    ) -> str:
        """
        Tool 2: Generate human-readable fingerprint behavior summary.
        
        Args:
            features: Feature vector from diagnostic run
            root_cause: Identified root cause
            
        Returns:
            Natural language feature description
        """
        descriptions = []
        
        # Vibration amplitude
        if features.rms_vibration > 4.0:
            descriptions.append(f"severe vibration intensity ({features.rms_vibration:.2f} m/s²)")
        elif features.rms_vibration > 3.0:
            descriptions.append(f"elevated vibration ({features.rms_vibration:.2f} m/s²)")
        else:
            descriptions.append(f"moderate vibration ({features.rms_vibration:.2f} m/s²)")
        
        # Frequency characteristics
        if features.dominant_frequency > 2000:
            descriptions.append(f"high-frequency signature at {features.dominant_frequency:.0f} Hz (bearing-range)")
        elif features.dominant_frequency > 200:
            descriptions.append(f"mid-frequency peak at {features.dominant_frequency:.0f} Hz")
        else:
            descriptions.append(f"low-frequency dominance at {features.dominant_frequency:.0f} Hz (structural/rotational)")
        
        # Spectral complexity
        if features.spectral_entropy > 0.85:
            descriptions.append(f"high spectral complexity (entropy {features.spectral_entropy:.2f}, indicating irregular/degraded components)")
        elif features.spectral_entropy > 0.65:
            descriptions.append(f"moderate spectral spread (entropy {features.spectral_entropy:.2f})")
        else:
            descriptions.append(f"clean frequency signature (entropy {features.spectral_entropy:.2f}, suggesting single-source excitation)")
        
        # Bearing signature
        if features.bearing_energy_band > 0.75:
            descriptions.append(f"strong bearing frequency energy ({features.bearing_energy_band:.2f}, indicating bearing surface damage)")
        elif features.bearing_energy_band > 0.60:
            descriptions.append(f"elevated bearing band energy ({features.bearing_energy_band:.2f})")
        
        # Audio anomaly
        if features.audio_anomaly_score > 0.80:
            descriptions.append(f"prominent audible anomaly (score {features.audio_anomaly_score:.2f})")
        
        # Speed dependency
        speed_desc = {
            "strong": "strongly speed-dependent (proportional to vehicle/rotation speed)",
            "medium": "moderately speed-dependent",
            "weak": "weakly speed-dependent (persistent across speeds)"
        }
        descriptions.append(speed_desc.get(features.speed_dependency, "speed dependency unknown"))
        
        # Build coherent summary
        summary = "Diagnostic fingerprint shows " + ", ".join(descriptions[:3])
        if len(descriptions) > 3:
            summary += ". Additional characteristics: " + ", ".join(descriptions[3:])
        
        summary += f". This pattern is consistent with {root_cause} behavior."
        
        return summary
    
    def summarize_causal_reasoning(
        self,
        root_cause: str,
        confidence: float,
        alternate_causes: Dict[str, float],
        fleet_matches: int,
        causal_reasoning: str
    ) -> str:
        """
        Tool 3: Generate summary of causal inference process.
        
        Args:
            root_cause: Identified root cause
            confidence: Confidence level
            alternate_causes: Alternate probable causes
            fleet_matches: Number of similar fleet cases
            causal_reasoning: Raw reasoning string from causal agent
            
        Returns:
            Natural language causal reasoning explanation
        """
        confidence_desc = {
            (0.9, 1.0): "very high confidence",
            (0.80, 0.90): "high confidence",
            (0.70, 0.80): "moderate-high confidence",
            (0.60, 0.70): "moderate confidence",
            (0.0, 0.60): "low-moderate confidence"
        }
        
        conf_label = next(
            (desc for (low, high), desc in confidence_desc.items() if low <= confidence < high),
            "moderate confidence"
        )
        
        summary = f"Causal analysis identifies **{root_cause}** as the root cause with {conf_label} ({confidence*100:.1f}%). "
        
        # Fleet context
        summary += f"This diagnosis is supported by comparison with {fleet_matches} similar historical fleet cases. "
        
        # Reasoning
        summary += f"Key diagnostic evidence: {causal_reasoning} "
        
        # Alternate causes
        if alternate_causes:
            top_alt = list(alternate_causes.items())[:2]
            alt_strs = [f"{cause} ({prob*100:.1f}%)" for cause, prob in top_alt]
            summary += f"Alternate possibilities considered: {', '.join(alt_strs)}. "
        
        # Confidence interpretation
        if confidence >= 0.75:
            summary += "Confidence level meets threshold for direct repair recommendation."
        else:
            summary += "Confidence level suggests active experiment for verification before repair commitment."
        
        return summary
    
    def generate_human_readable_report(
        self,
        current_run: DiagnosticRun,
        cause_json: Dict[str, Any],
        experiment_json: Dict[str, Any],
        repair_json: Dict[str, Any],
        fleet_summary: Dict[str, Any]
    ) -> str:
        """
        Tool 4: Generate complete human-readable diagnostic report.
        
        Args:
            current_run: Current diagnostic run
            cause_json: Causal inference results
            experiment_json: Experiment design results
            repair_json: Repair plan results
            fleet_summary: Fleet matching summary
            
        Returns:
            Complete natural language report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("MIRA WAVE DIAGNOSTIC REPORT")
        report_lines.append("Multi-Agent Reasoning & Intelligence for Automotive Analysis")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Run identification
        report_lines.append(f"Run ID: {current_run.run_id}")
        report_lines.append(f"Vehicle ID: {current_run.metadata.get('vehicle_id', 'Unknown')}")
        report_lines.append(f"Data Source: {current_run.metadata.get('sim_or_real', 'Unknown')}")
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("-" * 80)
        
        # Section 1: Fault Location
        report_lines.append("1. FAULT LOCALIZATION")
        report_lines.append("-" * 80)
        location_desc = self.summarize_fault_location(current_run.fault_location)
        report_lines.append(location_desc)
        report_lines.append("")
        
        # Section 2: Diagnostic Fingerprint
        report_lines.append("2. DIAGNOSTIC FINGERPRINT ANALYSIS")
        report_lines.append("-" * 80)
        fingerprint_desc = self.summarize_fingerprint_change(
            current_run.features,
            cause_json["root_cause"]
        )
        report_lines.append(fingerprint_desc)
        report_lines.append("")
        
        # Section 3: Fleet Correlation
        report_lines.append("3. FLEET PATTERN MATCHING")
        report_lines.append("-" * 80)
        fleet_confidence = fleet_summary.get("fleet_confidence", 0)
        avg_similarity = fleet_summary.get("avg_match_similarity", 0)
        report_lines.append(
            f"Current fault pattern matched against historical fleet database of 100+ vehicles. "
            f"Average similarity to top matches: {avg_similarity*100:.1f}%. "
            f"Fleet-based confidence: {fleet_confidence*100:.1f}%."
        )
        
        if "top_matches" in fleet_summary and fleet_summary["top_matches"]:
            report_lines.append("")
            report_lines.append("Top 3 Fleet Matches:")
            for i, match in enumerate(fleet_summary["top_matches"][:3], 1):
                similarity = match.get('similarity_score', match.get('similarity', 0))
                report_lines.append(
                    f"  {i}. {match['vehicle_id']}: {match['known_cause']} "
                    f"(similarity: {similarity*100:.1f}%)"
                )
        report_lines.append("")
        
        # Section 4: Causal Diagnosis
        report_lines.append("4. CAUSAL DIAGNOSIS")
        report_lines.append("-" * 80)
        causal_summary = self.summarize_causal_reasoning(
            cause_json["root_cause"],
            cause_json["confidence"],
            cause_json.get("alternate_causes", {}),
            len(fleet_summary.get("top_matches", [])),
            cause_json.get("reasoning", "")
        )
        report_lines.append(causal_summary)
        report_lines.append("")
        
        # Section 5: Active Experiment (if applicable)
        report_lines.append("5. ACTIVE EXPERIMENT RECOMMENDATION")
        report_lines.append("-" * 80)
        if experiment_json.get("experiment_required"):
            report_lines.append(
                f"**Experiment Required**: Confidence below threshold. "
                f"Recommended test: {experiment_json['recommended_experiment'].replace('_', ' ').title()}."
            )
            report_lines.append("")
            report_lines.append(f"Protocol: {experiment_json['instruction']}")
            report_lines.append("")
            report_lines.append(f"Expected Outcome: {experiment_json['expected_signal_change']}")
            report_lines.append("")
            report_lines.append(
                f"This experiment will reduce diagnostic uncertainty by approximately "
                f"{experiment_json['uncertainty_reduction_estimate']*100:.0f}%, enabling confident repair decisions."
            )
        else:
            reason = experiment_json.get("reason", "Confidence meets threshold")
            report_lines.append(f"**No Experiment Required**: {reason}")
            report_lines.append("Diagnostic confidence is sufficient to proceed directly to repair.")
        report_lines.append("")
        
        # Section 6: Repair Recommendation
        report_lines.append("6. REPAIR RECOMMENDATION")
        report_lines.append("-" * 80)
        report_lines.append(f"**Primary Action**: {repair_json['recommended_repair']}")
        report_lines.append("")
        report_lines.append(f"**Urgency Level**: {repair_json['urgency']}")
        
        if "urgency_details" in repair_json:
            reasons = repair_json["urgency_details"].get("reasons", [])
            if reasons:
                report_lines.append("Urgency Factors:")
                for reason in reasons:
                    report_lines.append(f"  - {reason}")
        
        report_lines.append("")
        report_lines.append(f"**Estimated Downtime**: {repair_json['estimated_downtime_hours']} hours")
        report_lines.append(f"**Complexity**: {repair_json.get('complexity', 'Medium').title()}")
        report_lines.append("")
        
        # Workshop requirements
        report_lines.append(f"**Recommended Workshop**: {repair_json['suggested_workshop_type']}")
        if "workshop_requirements" in repair_json:
            workshop = repair_json["workshop_requirements"]
            if workshop.get("certifications"):
                report_lines.append(f"Required Certifications: {', '.join(workshop['certifications'])}")
        
        report_lines.append("")
        
        # Cost estimate
        if "cost_estimate_range" in repair_json:
            cost = repair_json["cost_estimate_range"]
            report_lines.append(
                f"**Estimated Cost**: ${cost['min_usd']} - ${cost['max_usd']} USD "
                f"({cost.get('includes', 'parts and labor')})"
            )
        
        report_lines.append("")
        
        # Risk assessment
        report_lines.append("**Risk if Ignored**:")
        report_lines.append(repair_json.get("risk_if_ignored", "Progressive component degradation"))
        report_lines.append("")
        
        # Section 7: Detailed Steps (optional)
        if "detailed_steps" in repair_json:
            report_lines.append("7. DETAILED REPAIR STEPS")
            report_lines.append("-" * 80)
            for i, step in enumerate(repair_json["detailed_steps"], 1):
                report_lines.append(f"{i}. {step}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF DIAGNOSTIC REPORT")
        report_lines.append("")
        report_lines.append(
            "This report was generated by the MIRA Wave Person C multi-agent system, "
            "combining fleet-level pattern matching, Bayesian causal inference, active "
            "experiment design, and repair optimization."
        )
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_explanation_txt(
        self,
        report_text: str,
        run_id: str,
        output_dir: Path
    ) -> Path:
        """
        Tool 5: Save explanation report to text file.
        
        Args:
            report_text: Generated report text
            run_id: Run identifier
            output_dir: Directory to save output
            
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"explanation_{run_id}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return output_path
    
    def save_explanation_json(
        self,
        current_run: DiagnosticRun,
        cause_json: Dict[str, Any],
        experiment_json: Dict[str, Any],
        repair_json: Dict[str, Any],
        fleet_summary: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Generate structured JSON explanation (alternative to text).
        
        Args:
            current_run: Current diagnostic run
            cause_json: Causal inference results
            experiment_json: Experiment design results
            repair_json: Repair plan results
            fleet_summary: Fleet matching summary
            output_dir: Directory to save output
            
        Returns:
            Explanation JSON dictionary
        """
        explanation_json = {
            "run_id": current_run.run_id,
            "vehicle_id": current_run.metadata.get("vehicle_id", "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "fault_location": {
                "coordinates": current_run.fault_location.to_dict(),
                "description": self.summarize_fault_location(current_run.fault_location)
            },
            "fingerprint_summary": self.summarize_fingerprint_change(
                current_run.features,
                cause_json["root_cause"]
            ),
            "causal_summary": self.summarize_causal_reasoning(
                cause_json["root_cause"],
                cause_json["confidence"],
                cause_json.get("alternate_causes", {}),
                len(fleet_summary.get("top_matches", [])),
                cause_json.get("reasoning", "")
            ),
            "diagnosis": {
                "root_cause": cause_json["root_cause"],
                "confidence": cause_json["confidence"],
                "alternate_causes": cause_json.get("alternate_causes", {})
            },
            "experiment": {
                "required": experiment_json.get("experiment_required", False),
                "type": experiment_json.get("recommended_experiment"),
                "instruction": experiment_json.get("instruction")
            },
            "repair_plan": {
                "action": repair_json["recommended_repair"],
                "urgency": repair_json["urgency"],
                "downtime_hours": repair_json["estimated_downtime_hours"],
                "workshop": repair_json["suggested_workshop_type"],
                "risk": repair_json["risk_if_ignored"]
            },
            "fleet_context": {
                "matches_analyzed": len(fleet_summary.get("top_matches", [])),
                "average_similarity": fleet_summary.get("avg_match_similarity", 0),
                "fleet_confidence": fleet_summary.get("fleet_confidence", 0)
            }
        }
        
        # Save JSON
        try:
            from .utils import save_json
        except ImportError:
            from utils import save_json
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"explanation_{current_run.run_id}.json"
        save_json(explanation_json, output_path)
        
        return explanation_json
    
    def process_run(
        self,
        current_run: DiagnosticRun,
        cause_json: Dict[str, Any],
        experiment_json: Dict[str, Any],
        repair_json: Dict[str, Any],
        fleet_summary: Dict[str, Any],
        output_format: str = "txt"
    ) -> Dict[str, Any]:
        """
        Main processing method that orchestrates all explanation tools.
        
        Args:
            current_run: Current diagnostic run
            cause_json: Causal inference results
            experiment_json: Experiment design results
            repair_json: Repair plan results
            fleet_summary: Fleet matching summary
            output_format: "txt", "json", or "both"
            
        Returns:
            Explanation results with generated reports
        """
        # Generate location summary
        location_summary = self.summarize_fault_location(current_run.fault_location)
        
        # Generate fingerprint summary
        fingerprint_summary = self.summarize_fingerprint_change(
            current_run.features,
            cause_json["root_cause"]
        )
        
        # Generate causal summary
        causal_summary = self.summarize_causal_reasoning(
            cause_json["root_cause"],
            cause_json["confidence"],
            cause_json.get("alternate_causes", {}),
            len(fleet_summary.get("top_matches", [])),
            cause_json.get("reasoning", "")
        )
        
        results = {
            "run_id": current_run.run_id,
            "location_summary": location_summary,
            "fingerprint_summary": fingerprint_summary,
            "causal_summary": causal_summary
        }
        
        # Generate report(s) based on format
        if output_format in ["txt", "both"]:
            report_text = self.generate_human_readable_report(
                current_run,
                cause_json,
                experiment_json,
                repair_json,
                fleet_summary
            )
            results["report_text"] = report_text
        
        if output_format in ["json", "both"]:
            explanation_json = self.save_explanation_json(
                current_run,
                cause_json,
                experiment_json,
                repair_json,
                fleet_summary,
                Path("./output")
            )
            results["explanation_json"] = explanation_json
        
        return results
