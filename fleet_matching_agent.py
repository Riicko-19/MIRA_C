"""
Fleet Matching Agent for MIRA Wave Person C

Compares current diagnostic run against historical fleet data to identify
similar fault patterns through distance metrics, clustering, and statistical analysis.

Tools (10):
1. load_fleet_fingerprints()
2. compute_distance_matrix()
3. knn_match()
4. cluster_fingerprints()
5. compute_similarity_score()
6. retrieve_matched_runs()
7. summarize_cluster_statistics()
8. compute_centroid_embedding()
9. visualize_cluster_map()
10. save_matching_output()
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import json
from pathlib import Path

try:
    from .data_models import DiagnosticRun, FeatureVector, FaultLocation, FleetMatch, ClusterInfo
    from .utils import euclidean_distance, cosine_similarity, save_json
except ImportError:
    from data_models import DiagnosticRun, FeatureVector, FaultLocation, FleetMatch, ClusterInfo
    from utils import euclidean_distance, cosine_similarity, save_json


class FleetMatchingAgent:
    """
    Agent responsible for matching current fault fingerprints against
    historical fleet data to identify similar patterns and clusters.
    """
    
    def __init__(self, fleet_size: int = 100, random_seed: int = 42):
        """
        Initialize Fleet Matching Agent.
        
        Args:
            fleet_size: Number of synthetic fleet vehicles to generate
            random_seed: Random seed for reproducibility
        """
        self.fleet_size = fleet_size
        self.random_seed = random_seed
        self.fleet_fingerprints: List[Dict[str, Any]] = []
        self.distance_matrix: Optional[np.ndarray] = None
        self.clusters: Optional[List[ClusterInfo]] = None
        
        # Initialize synthetic fleet
        self._generate_synthetic_fleet()
    
    def _generate_synthetic_fleet(self) -> None:
        """Generate synthetic fleet history with diverse fault patterns."""
        np.random.seed(self.random_seed)
        
        # Define fault type templates
        fault_templates = {
            "Loose Mount": {
                "dominant_frequency": (80, 200),
                "rms_vibration": (3.0, 5.5),
                "spectral_entropy": (0.65, 0.85),
                "bearing_energy_band": (0.3, 0.6),
                "audio_anomaly_score": (0.6, 0.85),
                "speed_dependency": ["strong", "strong", "medium"]
            },
            "Bearing Wear": {
                "dominant_frequency": (2000, 3500),
                "rms_vibration": (1.5, 3.0),
                "spectral_entropy": (0.85, 0.98),
                "bearing_energy_band": (0.75, 0.95),
                "audio_anomaly_score": (0.7, 0.9),
                "speed_dependency": ["weak", "weak", "medium"]
            },
            "Imbalance": {
                "dominant_frequency": (50, 150),
                "rms_vibration": (2.0, 4.0),
                "spectral_entropy": (0.4, 0.65),
                "bearing_energy_band": (0.2, 0.5),
                "audio_anomaly_score": (0.5, 0.75),
                "speed_dependency": ["strong", "strong", "strong"]
            },
            "Misalignment": {
                "dominant_frequency": (70, 250),
                "rms_vibration": (2.5, 4.5),
                "spectral_entropy": (0.55, 0.75),
                "bearing_energy_band": (0.4, 0.7),
                "audio_anomaly_score": (0.55, 0.8),
                "speed_dependency": ["medium", "strong", "medium"]
            }
        }
        
        # Generate fleet vehicles with cause distribution
        cause_distribution = [
            ("Loose Mount", 35),
            ("Bearing Wear", 30),
            ("Imbalance", 20),
            ("Misalignment", 15)
        ]
        
        vehicle_id = 0
        for cause, count in cause_distribution:
            template = fault_templates[cause]
            
            for _ in range(count):
                # Generate features with variation
                features = FeatureVector(
                    dominant_frequency=np.random.uniform(*template["dominant_frequency"]),
                    rms_vibration=np.random.uniform(*template["rms_vibration"]),
                    spectral_entropy=np.random.uniform(*template["spectral_entropy"]),
                    bearing_energy_band=np.random.uniform(*template["bearing_energy_band"]),
                    audio_anomaly_score=np.random.uniform(*template["audio_anomaly_score"]),
                    speed_dependency=np.random.choice(template["speed_dependency"])
                )
                
                # Random fault location
                fault_location = FaultLocation(
                    x=np.random.uniform(0, 1),
                    y=np.random.uniform(0, 1)
                )
                
                self.fleet_fingerprints.append({
                    "vehicle_id": f"FLEET_{vehicle_id:04d}",
                    "run_id": f"HIST_{vehicle_id:04d}",
                    "features": features,
                    "fault_location": fault_location,
                    "known_cause": cause
                })
                
                vehicle_id += 1
    
    def load_fleet_fingerprints(self, filepath: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Tool 1: Load fleet fingerprints from file or return synthetic fleet.
        
        Args:
            filepath: Optional path to fleet data JSON file
            
        Returns:
            List of fleet fingerprint dictionaries
        """
        if filepath and filepath.exists():
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
                # Convert to proper data structures
                for item in loaded_data:
                    item["features"] = FeatureVector.from_dict(item["features"])
                    item["fault_location"] = FaultLocation.from_dict(item["fault_location"])
                self.fleet_fingerprints = loaded_data
        
        return self.fleet_fingerprints
    
    def compute_distance_matrix(
        self, 
        current_run: DiagnosticRun,
        metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Tool 2: Compute distance matrix between current run and all fleet vehicles.
        
        Args:
            current_run: Current diagnostic run to compare
            metric: Distance metric ("euclidean" or "cosine")
            
        Returns:
            Distance matrix as numpy array (1 x fleet_size)
        """
        current_vec = current_run.features.to_numeric_array()
        fleet_vecs = np.array([
            fp["features"].to_numeric_array() 
            for fp in self.fleet_fingerprints
        ])
        
        if metric == "euclidean":
            distances = pairwise_distances(
                current_vec.reshape(1, -1), 
                fleet_vecs, 
                metric='euclidean'
            )[0]
        elif metric == "cosine":
            # Convert to distance (1 - similarity)
            similarities = 1 - pairwise_distances(
                current_vec.reshape(1, -1), 
                fleet_vecs, 
                metric='cosine'
            )[0]
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.distance_matrix = distances
        return distances
    
    def knn_match(
        self, 
        current_run: DiagnosticRun, 
        k: int = 10,
        metric: str = "euclidean"
    ) -> List[FleetMatch]:
        """
        Tool 3: Find k-nearest neighbors in fleet history.
        
        Args:
            current_run: Current diagnostic run
            k: Number of nearest neighbors
            metric: Distance metric
            
        Returns:
            List of k closest FleetMatch objects
        """
        if self.distance_matrix is None:
            self.compute_distance_matrix(current_run, metric)
        
        # Get indices of k smallest distances
        k_indices = np.argsort(self.distance_matrix)[:k]
        
        matches = []
        for idx in k_indices:
            fp = self.fleet_fingerprints[idx]
            similarity = 1.0 / (1.0 + self.distance_matrix[idx])  # Convert distance to similarity
            
            matches.append(FleetMatch(
                vehicle_id=fp["vehicle_id"],
                run_id=fp["run_id"],
                similarity_score=similarity,
                distance=self.distance_matrix[idx],
                known_cause=fp["known_cause"],
                features=fp["features"],
                fault_location=fp["fault_location"]
            ))
        
        return matches
    
    def cluster_fingerprints(
        self, 
        current_run: DiagnosticRun,
        n_clusters: int = 4
    ) -> Tuple[int, List[ClusterInfo]]:
        """
        Tool 4: Cluster fleet fingerprints and identify which cluster current run belongs to.
        
        Args:
            current_run: Current diagnostic run
            n_clusters: Number of clusters to create
            
        Returns:
            Tuple of (assigned_cluster_id, list_of_cluster_info)
        """
        # Prepare feature matrix
        fleet_vecs = np.array([
            fp["features"].to_numeric_array() 
            for fp in self.fleet_fingerprints
        ])
        current_vec = current_run.features.to_numeric_array().reshape(1, -1)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(fleet_vecs)
        
        # Assign current run to nearest cluster
        current_cluster = kmeans.predict(current_vec)[0]
        
        # Build cluster information
        cluster_info_list = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_members = [
                self.fleet_fingerprints[i] 
                for i in range(len(self.fleet_fingerprints)) 
                if cluster_mask[i]
            ]
            
            # Compute cause distribution in cluster
            cause_counts = {}
            for member in cluster_members:
                cause = member["known_cause"]
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
            
            total_members = len(cluster_members)
            cause_distribution = {
                cause: count / total_members 
                for cause, count in cause_counts.items()
            }
            
            dominant_cause = max(cause_counts, key=cause_counts.get)
            
            # Average similarity within cluster
            cluster_vecs = fleet_vecs[cluster_mask]
            if len(cluster_vecs) > 1:
                intra_distances = pairwise_distances(cluster_vecs, metric='euclidean')
                avg_similarity = 1.0 / (1.0 + np.mean(intra_distances))
            else:
                avg_similarity = 1.0
            
            cluster_info_list.append(ClusterInfo(
                cluster_id=cluster_id,
                centroid=kmeans.cluster_centers_[cluster_id],
                member_count=total_members,
                dominant_cause=dominant_cause,
                cause_distribution=cause_distribution,
                avg_similarity=avg_similarity
            ))
        
        self.clusters = cluster_info_list
        return current_cluster, cluster_info_list
    
    def compute_similarity_score(
        self, 
        current_run: DiagnosticRun, 
        fleet_vehicle: Dict[str, Any]
    ) -> float:
        """
        Tool 5: Compute detailed similarity score between current run and a fleet vehicle.
        
        Args:
            current_run: Current diagnostic run
            fleet_vehicle: Fleet vehicle dictionary
            
        Returns:
            Similarity score (0 to 1)
        """
        current_vec = current_run.features.to_numeric_array()
        fleet_vec = fleet_vehicle["features"].to_numeric_array()
        
        # Use exponential decay distance-to-similarity conversion
        distance = euclidean_distance(current_vec, fleet_vec)
        similarity = np.exp(-distance)
        
        return float(similarity)
    
    def retrieve_matched_runs(
        self, 
        matches: List[FleetMatch]
    ) -> List[Dict[str, Any]]:
        """
        Tool 6: Retrieve full details of matched runs.
        
        Args:
            matches: List of FleetMatch objects from knn_match
            
        Returns:
            List of detailed match dictionaries
        """
        detailed_matches = []
        for match in matches:
            detailed_matches.append({
                "vehicle_id": match.vehicle_id,
                "run_id": match.run_id,
                "similarity_score": match.similarity_score,
                "distance": match.distance,
                "known_cause": match.known_cause,
                "features": match.features.to_dict(),
                "fault_location": match.fault_location.to_dict()
            })
        
        return detailed_matches
    
    def summarize_cluster_statistics(
        self, 
        cluster_info: ClusterInfo
    ) -> Dict[str, Any]:
        """
        Tool 7: Generate statistical summary of a cluster.
        
        Args:
            cluster_info: ClusterInfo object to summarize
            
        Returns:
            Dictionary with cluster statistics
        """
        return {
            "cluster_id": cluster_info.cluster_id,
            "member_count": cluster_info.member_count,
            "dominant_cause": cluster_info.dominant_cause,
            "cause_distribution": cluster_info.cause_distribution,
            "avg_intra_cluster_similarity": cluster_info.avg_similarity,
            "confidence": max(cluster_info.cause_distribution.values())
        }
    
    def compute_centroid_embedding(
        self, 
        cluster_info_list: List[ClusterInfo]
    ) -> np.ndarray:
        """
        Tool 8: Compute centroid embeddings for all clusters.
        
        Args:
            cluster_info_list: List of ClusterInfo objects
            
        Returns:
            Matrix of centroid embeddings (n_clusters x feature_dim)
        """
        centroids = np.array([
            cluster.centroid 
            for cluster in cluster_info_list
        ])
        return centroids
    
    def visualize_cluster_map(
        self, 
        current_run: DiagnosticRun,
        cluster_info_list: List[ClusterInfo]
    ) -> Dict[str, Any]:
        """
        Tool 9: Generate visualization-ready data structure for cluster map.
        
        Args:
            current_run: Current diagnostic run
            cluster_info_list: List of ClusterInfo objects
            
        Returns:
            Dictionary containing visualization data
        """
        # Use first 2 principal dimensions of features for 2D visualization
        fleet_vecs = np.array([
            fp["features"].to_numeric_array() 
            for fp in self.fleet_fingerprints
        ])
        
        # Simple 2D projection using first two feature dimensions
        projection_2d = fleet_vecs[:, :2]
        current_projection = current_run.features.to_numeric_array()[:2]
        
        # Organize by cluster
        cluster_points = {}
        for idx, fp in enumerate(self.fleet_fingerprints):
            # Find which cluster this point belongs to
            for cluster in cluster_info_list:
                cluster_id = cluster.cluster_id
                if cluster_id not in cluster_points:
                    cluster_points[cluster_id] = {
                        "points": [],
                        "dominant_cause": cluster.dominant_cause
                    }
        
        # Group points by actual cluster assignment (recompute for visualization)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(cluster_info_list), random_state=self.random_seed, n_init=10)
        labels = kmeans.fit_predict(fleet_vecs)
        
        for idx, label in enumerate(labels):
            cluster_points[label]["points"].append({
                "x": float(projection_2d[idx, 0]),
                "y": float(projection_2d[idx, 1]),
                "cause": self.fleet_fingerprints[idx]["known_cause"]
            })
        
        return {
            "current_run_position": {
                "x": float(current_projection[0]),
                "y": float(current_projection[1])
            },
            "clusters": cluster_points
        }
    
    def save_matching_output(
        self, 
        current_run: DiagnosticRun,
        matches: List[FleetMatch],
        assigned_cluster: int,
        cluster_info_list: List[ClusterInfo],
        output_dir: Path
    ) -> None:
        """
        Tool 10: Save fleet matching results to JSON file.
        
        Args:
            current_run: Current diagnostic run
            matches: List of matched fleet vehicles
            assigned_cluster: Cluster ID for current run
            cluster_info_list: All cluster information
            output_dir: Directory to save output
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find current cluster info
        current_cluster_info = next(
            (c for c in cluster_info_list if c.cluster_id == assigned_cluster),
            None
        )
        
        output_data = {
            "run_id": current_run.run_id,
            "assigned_cluster": assigned_cluster,
            "cluster_summary": self.summarize_cluster_statistics(current_cluster_info) if current_cluster_info else {},
            "top_matches": [
                {
                    "vehicle_id": m.vehicle_id,
                    "similarity": m.similarity_score,
                    "known_cause": m.known_cause
                }
                for m in matches[:5]
            ],
            "fleet_cause_votes": {}
        }
        
        # Count causes in top matches
        for match in matches:
            cause = match.known_cause
            output_data["fleet_cause_votes"][cause] = output_data["fleet_cause_votes"].get(cause, 0) + 1
        
        # Normalize to probabilities
        total_votes = sum(output_data["fleet_cause_votes"].values())
        output_data["fleet_cause_votes"] = {
            cause: count / total_votes
            for cause, count in output_data["fleet_cause_votes"].items()
        }
        
        output_path = output_dir / f"fleet_matching_{current_run.run_id}.json"
        save_json(output_data, output_path)
    
    def process_run(
        self, 
        current_run: DiagnosticRun,
        k_neighbors: int = 15,
        n_clusters: int = 4
    ) -> Dict[str, Any]:
        """
        Main processing method that orchestrates all tools for a single run.
        
        Args:
            current_run: Current diagnostic run
            k_neighbors: Number of neighbors for KNN matching
            n_clusters: Number of clusters for clustering
            
        Returns:
            Comprehensive fleet matching results dictionary
        """
        # Execute tool pipeline
        self.load_fleet_fingerprints()
        self.compute_distance_matrix(current_run)
        matches = self.knn_match(current_run, k=k_neighbors)
        assigned_cluster, cluster_info_list = self.cluster_fingerprints(current_run, n_clusters)
        
        # Get current cluster info
        current_cluster = next(
            (c for c in cluster_info_list if c.cluster_id == assigned_cluster),
            None
        )
        
        # Aggregate cause probabilities from matches
        cause_votes = {}
        for match in matches:
            cause = match.known_cause
            # Weight by similarity
            cause_votes[cause] = cause_votes.get(cause, 0) + match.similarity_score
        
        # Normalize
        total_weight = sum(cause_votes.values())
        if total_weight > 0:
            cause_probabilities = {
                cause: weight / total_weight 
                for cause, weight in cause_votes.items()
            }
        else:
            cause_probabilities = {}
        
        return {
            "run_id": current_run.run_id,
            "assigned_cluster": assigned_cluster,
            "cluster_info": self.summarize_cluster_statistics(current_cluster) if current_cluster else {},
            "top_matches": self.retrieve_matched_runs(matches[:10]),
            "cause_probabilities_from_fleet": cause_probabilities,
            "avg_match_similarity": np.mean([m.similarity_score for m in matches]),
            "fleet_confidence": max(cause_probabilities.values()) if cause_probabilities else 0.0
        }
