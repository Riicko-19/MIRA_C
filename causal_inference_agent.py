"""
Causal Inference Agent for MIRA Wave Person C

Estimates probabilistic fault attribution using Bayesian reasoning,
feature correlations, and fleet-informed causal models.

Tools (8):
1. compute_feature_correlations()
2. estimate_treatment_effect()
3. compute_causal_graph()
4. bayesian_cause_posterior()
5. rank_cause_probabilities()
6. compute_confidence_interval()
7. compare_before_after_fingerprints()
8. export_causal_json()
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
from pathlib import Path

try:
    from .data_models import DiagnosticRun, FeatureVector, FleetMatch
    from .utils import (
        normalize_probabilities, save_json, safe_corrcoef, ensure_finite,
        get_top_probability, compute_probability_confidence
    )
except ImportError:
    from data_models import DiagnosticRun, FeatureVector, FleetMatch
    from utils import (
        normalize_probabilities, save_json, safe_corrcoef, ensure_finite,
        get_top_probability, compute_probability_confidence
    )


class CausalInferenceAgent:
    """
    Agent responsible for computing probabilistic fault attribution
    using causal reasoning over features and fleet patterns.
    """
    
    def __init__(self):
        """Initialize Causal Inference Agent with fault signature knowledge."""
        
        # Define causal fault signatures (prior knowledge)
        self.fault_signatures = {
            "Loose Mount": {
                "rms_vibration": {"mean": 4.25, "std": 0.75, "weight": 0.25},
                "dominant_frequency": {"mean": 140, "std": 50, "weight": 0.15},
                "spectral_entropy": {"mean": 0.75, "std": 0.08, "weight": 0.15},
                "bearing_energy_band": {"mean": 0.45, "std": 0.12, "weight": 0.10},
                "audio_anomaly_score": {"mean": 0.725, "std": 0.10, "weight": 0.10},
                "speed_dependency": {"strong": 0.75, "medium": 0.20, "weak": 0.05, "weight": 0.25}
            },
            "Bearing Wear": {
                "rms_vibration": {"mean": 2.25, "std": 0.60, "weight": 0.15},
                "dominant_frequency": {"mean": 2750, "std": 600, "weight": 0.20},
                "spectral_entropy": {"mean": 0.915, "std": 0.05, "weight": 0.25},
                "bearing_energy_band": {"mean": 0.85, "std": 0.08, "weight": 0.25},
                "audio_anomaly_score": {"mean": 0.80, "std": 0.08, "weight": 0.10},
                "speed_dependency": {"strong": 0.10, "medium": 0.30, "weak": 0.60, "weight": 0.05}
            },
            "Imbalance": {
                "rms_vibration": {"mean": 3.0, "std": 0.70, "weight": 0.20},
                "dominant_frequency": {"mean": 100, "std": 40, "weight": 0.20},
                "spectral_entropy": {"mean": 0.525, "std": 0.10, "weight": 0.20},
                "bearing_energy_band": {"mean": 0.35, "std": 0.12, "weight": 0.10},
                "audio_anomaly_score": {"mean": 0.625, "std": 0.10, "weight": 0.10},
                "speed_dependency": {"strong": 0.85, "medium": 0.12, "weak": 0.03, "weight": 0.20}
            },
            "Misalignment": {
                "rms_vibration": {"mean": 3.5, "std": 0.70, "weight": 0.20},
                "dominant_frequency": {"mean": 160, "std": 70, "weight": 0.15},
                "spectral_entropy": {"mean": 0.65, "std": 0.08, "weight": 0.15},
                "bearing_energy_band": {"mean": 0.55, "std": 0.12, "weight": 0.15},
                "audio_anomaly_score": {"mean": 0.675, "std": 0.10, "weight": 0.10},
                "speed_dependency": {"strong": 0.50, "medium": 0.40, "weak": 0.10, "weight": 0.25}
            }
        }
        
        # Base prior probabilities (uniform initially)
        self.prior_probabilities = {
            "Loose Mount": 0.25,
            "Bearing Wear": 0.25,
            "Imbalance": 0.25,
            "Misalignment": 0.25
        }
        
        self.feature_correlations: Optional[Dict[str, Any]] = None
        self.causal_graph: Optional[Dict[str, List[str]]] = None
    
    def compute_feature_correlations(
        self, 
        current_run: DiagnosticRun,
        fleet_matches: List[FleetMatch]
    ) -> Dict[str, float]:
        """
        Tool 1: Compute feature correlations between current run and fleet matches.
        
        Args:
            current_run: Current diagnostic run
            fleet_matches: List of matched fleet vehicles
            
        Returns:
            Dictionary of correlation coefficients per feature
        """
        if not fleet_matches:
            return {}
        
        current_features = current_run.features.to_numeric_array()
        match_features = np.array([
            match.features.to_numeric_array() 
            for match in fleet_matches
        ])
        
        # Compute correlation for each feature dimension
        correlations = {}
        feature_names = [
            "dominant_frequency", "rms_vibration", "spectral_entropy",
            "bearing_energy_band", "audio_anomaly_score", "speed_dependency"
        ]
        
        for i, feature_name in enumerate(feature_names):
            if len(match_features) > 1:
                # Pearson correlation between current value and fleet distribution
                # Using safe_corrcoef to avoid RuntimeWarnings when feature variance is zero
                correlation = safe_corrcoef(
                    np.append(match_features[:, i], current_features[i]),
                    np.append(np.zeros(len(match_features)), 1)
                )
                correlations[feature_name] = correlation
            else:
                correlations[feature_name] = 0.0
        
        self.feature_correlations = correlations
        return correlations
    
    def estimate_treatment_effect(
        self,
        current_run: DiagnosticRun,
        fleet_matches: List[FleetMatch],
        cause: str
    ) -> float:
        """
        Tool 2: Estimate treatment effect (causal influence) of a hypothesized cause.
        
        Args:
            current_run: Current diagnostic run
            fleet_matches: List of matched fleet vehicles
            cause: Hypothesized cause to estimate effect for
            
        Returns:
            Treatment effect magnitude (0 to 1)
        """
        # Filter matches by cause
        cause_matches = [m for m in fleet_matches if m.known_cause == cause]
        other_matches = [m for m in fleet_matches if m.known_cause != cause]
        
        if not cause_matches:
            return 0.0
        
        current_vec = current_run.features.to_numeric_array()
        
        # Compute average distance to cause matches vs other matches
        cause_distances = [
            np.linalg.norm(current_vec - m.features.to_numeric_array())
            for m in cause_matches
        ]
        
        if other_matches:
            other_distances = [
                np.linalg.norm(current_vec - m.features.to_numeric_array())
                for m in other_matches
            ]
            avg_other_dist = np.mean(other_distances)
        else:
            avg_other_dist = 1.0
        
        avg_cause_dist = np.mean(cause_distances)
        
        # Treatment effect: relative closeness to cause vs others
        if avg_other_dist > 0:
            effect = max(0, 1 - (avg_cause_dist / avg_other_dist))
        else:
            effect = 0.5
        
        return float(np.clip(effect, 0, 1))
    
    def compute_causal_graph(
        self,
        current_run: DiagnosticRun
    ) -> Dict[str, List[str]]:
        """
        Tool 3: Compute lightweight causal graph showing feature dependencies.
        
        Args:
            current_run: Current diagnostic run
            
        Returns:
            Adjacency list representing causal dependencies
        """
        # Simplified causal graph based on automotive dynamics knowledge
        causal_graph = {
            "root_cause": ["mechanical_state"],
            "mechanical_state": ["vibration_signature", "acoustic_signature"],
            "vibration_signature": ["rms_vibration", "dominant_frequency", "speed_dependency"],
            "acoustic_signature": ["audio_anomaly_score", "spectral_entropy"],
            "bearing_condition": ["bearing_energy_band", "spectral_entropy"],
            "structural_integrity": ["rms_vibration", "speed_dependency"]
        }
        
        self.causal_graph = causal_graph
        return causal_graph
    
    def bayesian_cause_posterior(
        self,
        current_run: DiagnosticRun,
        fleet_cause_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Tool 4: Compute Bayesian posterior probabilities for each cause.
        
        Uses feature likelihoods and fleet-informed priors.
        Phase 3 enhanced: ensures posteriors are always finite and normalized.
        
        Args:
            current_run: Current diagnostic run
            fleet_cause_probs: Prior probabilities from fleet matching
            
        Returns:
            Posterior probability distribution over causes (guaranteed to sum to 1.0)
        """
        features = current_run.features
        posteriors = {}
        
        # Compute likelihood for each cause
        for cause, signature in self.fault_signatures.items():
            likelihood = 1.0
            
            # Numeric features (Gaussian likelihood)
            numeric_features = {
                "rms_vibration": features.rms_vibration,
                "dominant_frequency": features.dominant_frequency,
                "spectral_entropy": features.spectral_entropy,
                "bearing_energy_band": features.bearing_energy_band,
                "audio_anomaly_score": features.audio_anomaly_score
            }
            
            for feature_name, feature_value in numeric_features.items():
                if feature_name in signature:
                    sig = signature[feature_name]
                    mean, std, weight = sig["mean"], sig["std"], sig["weight"]
                    
                    # Gaussian probability density (with safety checks)
                    prob = stats.norm.pdf(feature_value, mean, std)
                    max_prob = stats.norm.pdf(mean, mean, std)
                    
                    # Normalize to 0-1 range (with division safety)
                    if max_prob > 1e-10:
                        normalized_prob = min(prob / max_prob, 1.0)
                    else:
                        normalized_prob = 0.5  # Neutral if signature std is degenerate
                    
                    # Ensure finite
                    normalized_prob = ensure_finite(normalized_prob, 0.5)
                    
                    # Apply weighted contribution (multiplicative model)
                    likelihood *= (1 - weight + weight * normalized_prob)
            
            # Categorical feature: speed_dependency
            if "speed_dependency" in signature:
                speed_sig = signature["speed_dependency"]
                speed_prob = speed_sig.get(features.speed_dependency, 0.1)
                speed_weight = speed_sig["weight"]
                likelihood *= (1 - speed_weight + speed_weight * speed_prob)
            
            # Ensure likelihood is finite and positive
            likelihood = max(ensure_finite(likelihood, 0.01), 1e-6)
            
            # Combine with fleet prior
            fleet_prior = fleet_cause_probs.get(cause, 0.25)
            fleet_prior = max(ensure_finite(fleet_prior, 0.25), 0.01)
            
            base_prior = self.prior_probabilities[cause]
            
            # Weighted average of base prior and fleet prior
            combined_prior = 0.3 * base_prior + 0.7 * fleet_prior
            
            # Posterior = likelihood × prior
            posteriors[cause] = likelihood * combined_prior
        
        # Normalize to sum to 1 (using Phase 3 robust normalization)
        posteriors = normalize_probabilities(posteriors)
        
        return posteriors
    
    def rank_cause_probabilities(
        self,
        posteriors: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Tool 5: Rank causes by posterior probability.
        
        Args:
            posteriors: Posterior probability distribution
            
        Returns:
            List of (cause, probability) tuples sorted by probability (descending)
        """
        ranked = sorted(
            posteriors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
    
    def compute_confidence_interval(
        self,
        posteriors: Dict[str, float],
        fleet_matches: List[FleetMatch],
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Tool 6: Compute confidence intervals for cause probabilities.
        
        Args:
            posteriors: Posterior probability distribution
            fleet_matches: Fleet matches for bootstrapping
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary mapping causes to (lower_bound, upper_bound) tuples
        """
        if not fleet_matches or len(fleet_matches) < 3:
            # Not enough data for meaningful CI
            return {
                cause: (prob * 0.8, min(prob * 1.2, 1.0))
                for cause, prob in posteriors.items()
            }
        
        # Bootstrap-based confidence intervals
        n_bootstrap = 1000
        bootstrap_probs = {cause: [] for cause in posteriors.keys()}
        
        for _ in range(n_bootstrap):
            # Resample matches with replacement
            resampled = np.random.choice(fleet_matches, size=len(fleet_matches), replace=True)
            
            # Compute cause distribution in resample
            cause_counts = {}
            for match in resampled:
                cause = match.known_cause
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
            
            total = len(resampled)
            for cause in posteriors.keys():
                prob = cause_counts.get(cause, 0) / total
                bootstrap_probs[cause].append(prob)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        intervals = {}
        
        for cause, probs in bootstrap_probs.items():
            probs_array = np.array(probs)
            lower = np.percentile(probs_array, alpha / 2 * 100)
            upper = np.percentile(probs_array, (1 - alpha / 2) * 100)
            intervals[cause] = (float(lower), float(upper))
        
        return intervals
    
    def compare_before_after_fingerprints(
        self,
        before_run: DiagnosticRun,
        after_run: DiagnosticRun
    ) -> Dict[str, Any]:
        """
        Tool 7: Compare fingerprints before and after an intervention/experiment.
        
        Args:
            before_run: Diagnostic run before intervention
            after_run: Diagnostic run after intervention
            
        Returns:
            Dictionary of feature changes and significance
        """
        before_vec = before_run.features.to_numeric_array()
        after_vec = after_run.features.to_numeric_array()
        
        feature_names = [
            "dominant_frequency", "rms_vibration", "spectral_entropy",
            "bearing_energy_band", "audio_anomaly_score", "speed_dependency"
        ]
        
        changes = {}
        for i, feature_name in enumerate(feature_names):
            before_val = before_vec[i]
            after_val = after_vec[i]
            
            absolute_change = after_val - before_val
            if before_val != 0:
                relative_change = (after_val - before_val) / before_val
            else:
                relative_change = 0.0
            
            changes[feature_name] = {
                "before": float(before_val),
                "after": float(after_val),
                "absolute_change": float(absolute_change),
                "relative_change": float(relative_change),
                "significant": abs(relative_change) > 0.15  # 15% threshold
            }
        
        return changes
    
    def export_causal_json(
        self,
        current_run: DiagnosticRun,
        posteriors: Dict[str, float],
        ranked_causes: List[Tuple[str, float]],
        confidence_intervals: Dict[str, Tuple[float, float]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Tool 8: Export causal inference results to cause.json format.
        
        Phase 3 enhanced: guarantees all probabilities are finite and valid.
        
        Args:
            current_run: Current diagnostic run
            posteriors: Posterior probability distribution
            ranked_causes: Ranked list of causes
            confidence_intervals: Confidence intervals for each cause
            output_dir: Directory to save output
            
        Returns:
            Dictionary in cause.json format with validated probabilities
        """
        # Use safe extraction (handles empty list gracefully)
        if ranked_causes:
            root_cause, confidence = ranked_causes[0]
        else:
            # Fallback if ranking failed
            root_cause, confidence = get_top_probability(posteriors)
        
        # Ensure confidence is in valid range
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Build alternate causes (exclude root cause)
        alternate_causes = {
            cause: float(np.clip(prob, 0.0, 1.0))
            for cause, prob in ranked_causes[1:4]  # Top 3 alternates
            if cause != root_cause
        }
        
        # Generate reasoning based on features
        features = current_run.features
        reasoning_parts = []
        
        # Analyze key features
        if features.rms_vibration > 3.5:
            reasoning_parts.append(f"High RMS vibration ({features.rms_vibration:.2f})")
        
        if features.dominant_frequency > 2000:
            reasoning_parts.append(f"high-frequency dominance at {features.dominant_frequency:.1f} Hz")
        elif features.dominant_frequency < 200:
            reasoning_parts.append(f"low-frequency dominance at {features.dominant_frequency:.1f} Hz")
        
        if features.spectral_entropy > 0.85:
            reasoning_parts.append(f"elevated spectral entropy ({features.spectral_entropy:.2f})")
        
        if features.bearing_energy_band > 0.75:
            reasoning_parts.append(f"strong bearing energy signature ({features.bearing_energy_band:.2f})")
        
        if features.speed_dependency == "strong":
            reasoning_parts.append("strong speed dependency")
        
        reasoning = " combined with ".join(reasoning_parts[:3]) if reasoning_parts else "Feature analysis"
        reasoning += f" matches {root_cause} behavior with {confidence*100:.0f}% confidence based on fleet correlation and causal signature analysis."
        
        # Build confidence interval (with safety checks)
        ci_lower, ci_upper = confidence_intervals.get(root_cause, (confidence * 0.8, min(confidence * 1.2, 1.0)))
        ci_lower = float(np.clip(ci_lower, 0.0, 1.0))
        ci_upper = float(np.clip(ci_upper, 0.0, 1.0))
        
        cause_json = {
            "run_id": current_run.run_id,
            "root_cause": root_cause,
            "confidence": round(confidence, 3),
            "alternate_causes": {k: round(v, 3) for k, v in alternate_causes.items()},
            "reasoning": reasoning,
            "confidence_interval": {
                "lower": round(ci_lower, 3),
                "upper": round(ci_upper, 3)
            }
        }
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cause_{current_run.run_id}.json"
        save_json(cause_json, output_path)
        
        return cause_json
    
    def process_run(
        self,
        current_run: DiagnosticRun,
        fleet_matches: List[FleetMatch],
        fleet_cause_probs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Main processing method that orchestrates all causal inference tools.
        
        Args:
            current_run: Current diagnostic run
            fleet_matches: Matched fleet vehicles from FleetMatchingAgent
            fleet_cause_probs: Cause probabilities from fleet matching
            
        Returns:
            Comprehensive causal inference results
        """
        # Execute tool pipeline
        correlations = self.compute_feature_correlations(current_run, fleet_matches)
        causal_graph = self.compute_causal_graph(current_run)
        posteriors = self.bayesian_cause_posterior(current_run, fleet_cause_probs)
        ranked_causes = self.rank_cause_probabilities(posteriors)
        confidence_intervals = self.compute_confidence_interval(posteriors, fleet_matches)
        
        # Compute treatment effects for all causes
        treatment_effects = {}
        for cause in self.fault_signatures.keys():
            treatment_effects[cause] = self.estimate_treatment_effect(
                current_run, fleet_matches, cause
            )
        
        root_cause, confidence = ranked_causes[0]
        
        return {
            "run_id": current_run.run_id,
            "root_cause": root_cause,
            "confidence": confidence,
            "posteriors": posteriors,
            "ranked_causes": ranked_causes,
            "confidence_intervals": confidence_intervals,
            "treatment_effects": treatment_effects,
            "feature_correlations": correlations,
            "causal_graph": causal_graph
        }
