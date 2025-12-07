"""
Data models and structures for MIRA Wave Person C diagnostic system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FaultLocation:
    """Represents a fault location on a 2D virtual vehicle map."""
    x: float
    y: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'FaultLocation':
        return cls(x=data["x"], y=data["y"])


@dataclass
class FeatureVector:
    """Represents extracted vehicle fault fingerprint features."""
    dominant_frequency: float
    rms_vibration: float
    spectral_entropy: float
    bearing_energy_band: float
    audio_anomaly_score: float
    speed_dependency: str  # "weak", "medium", "strong"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_frequency": self.dominant_frequency,
            "rms_vibration": self.rms_vibration,
            "spectral_entropy": self.spectral_entropy,
            "bearing_energy_band": self.bearing_energy_band,
            "audio_anomaly_score": self.audio_anomaly_score,
            "speed_dependency": self.speed_dependency,
        }
    
    def to_numeric_array(self) -> np.ndarray:
        """Convert to numeric array for distance calculations."""
        speed_map = {"weak": 0.33, "medium": 0.67, "strong": 1.0}
        return np.array([
            self.dominant_frequency / 3000.0,  # Normalize to ~0-1
            self.rms_vibration / 10.0,
            self.spectral_entropy,
            self.bearing_energy_band,
            self.audio_anomaly_score,
            speed_map.get(self.speed_dependency, 0.67)
        ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVector':
        return cls(
            dominant_frequency=data["dominant_frequency"],
            rms_vibration=data["rms_vibration"],
            spectral_entropy=data["spectral_entropy"],
            bearing_energy_band=data["bearing_energy_band"],
            audio_anomaly_score=data["audio_anomaly_score"],
            speed_dependency=data["speed_dependency"]
        )


@dataclass
class DiagnosticRun:
    """Represents a single diagnostic run input."""
    run_id: str
    fault_location: FaultLocation
    features: FeatureVector
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiagnosticRun':
        return cls(
            run_id=data["run_id"],
            fault_location=FaultLocation.from_dict(data["fault_location"]),
            features=FeatureVector.from_dict(data["features"]),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fault_location": self.fault_location.to_dict(),
            "features": self.features.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class FleetMatch:
    """Represents a matched historical fleet vehicle."""
    vehicle_id: str
    run_id: str
    similarity_score: float
    distance: float
    known_cause: str
    features: FeatureVector
    fault_location: FaultLocation


@dataclass
class ClusterInfo:
    """Represents a cluster of similar fault patterns."""
    cluster_id: int
    centroid: np.ndarray
    member_count: int
    dominant_cause: str
    cause_distribution: Dict[str, float]
    avg_similarity: float
