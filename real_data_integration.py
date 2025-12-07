"""
Real Fleet Database Loader for MIRA Wave Person C

Demonstrates how to integrate real historical fleet data instead of synthetic data.
"""

import json
from pathlib import Path
from fleet_matching_agent import FleetMatchingAgent
from data_models import FeatureVector, FaultLocation


def load_real_fleet_from_database(database_path: Path) -> list:
    """
    Load real fleet history from a database or file.
    
    Expected format:
    [
        {
            "vehicle_id": "VIN_12345",
            "run_id": "HIST_001",
            "features": {
                "dominant_frequency": 128.5,
                "rms_vibration": 3.8,
                ...
            },
            "fault_location": {"x": 0.7, "y": 0.3},
            "known_cause": "Loose Mount",
            "repair_confirmed": true,
            "timestamp": "2025-01-15T14:30:00Z"
        },
        ...
    ]
    """
    if not database_path.exists():
        print(f"Real fleet database not found at {database_path}")
        print("Falling back to synthetic fleet for demonstration.")
        return None
    
    with open(database_path, 'r') as f:
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


def create_fleet_agent_with_real_data(fleet_database_path: Path = None):
    """
    Create FleetMatchingAgent with real fleet data if available.
    
    Usage:
        fleet_agent = create_fleet_agent_with_real_data(
            Path("./real_fleet_history.json")
        )
    """
    agent = FleetMatchingAgent(fleet_size=100)
    
    if fleet_database_path:
        real_fleet = load_real_fleet_from_database(fleet_database_path)
        if real_fleet:
            agent.fleet_fingerprints = real_fleet
            print(f"Loaded {len(real_fleet)} real fleet cases")
        else:
            print("Using synthetic fleet data")
    
    return agent


# Example: Creating a real fleet database from repair shop records
def convert_repair_records_to_fleet_database(
    repair_records_path: Path,
    output_path: Path
):
    """
    Convert repair shop records into fleet database format.
    
    This is a template - adapt to your actual data source format.
    """
    # Example: Read from CSV, database, or API
    import pandas as pd
    
    # Load repair records (adapt to your format)
    # df = pd.read_csv(repair_records_path)
    
    fleet_database = []
    
    # Example conversion (customize based on your data schema)
    # for idx, row in df.iterrows():
    #     fleet_database.append({
    #         "vehicle_id": row["vin"],
    #         "run_id": f"HIST_{idx:05d}",
    #         "features": {
    #             "dominant_frequency": row["freq_hz"],
    #             "rms_vibration": row["vib_rms"],
    #             "spectral_entropy": row["entropy"],
    #             "bearing_energy_band": row["bearing_energy"],
    #             "audio_anomaly_score": row["audio_score"],
    #             "speed_dependency": row["speed_dep"]
    #         },
    #         "fault_location": {
    #             "x": row["loc_x"],
    #             "y": row["loc_y"]
    #         },
    #         "known_cause": row["confirmed_fault"],
    #         "repair_confirmed": True,
    #         "timestamp": row["repair_date"]
    #     })
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(fleet_database, f, indent=2)
    
    print(f"Fleet database saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    fleet_db_path = Path("./real_fleet_history.json")
    
    # Option 1: Load existing real fleet
    agent = create_fleet_agent_with_real_data(fleet_db_path)
    
    # Option 2: Convert repair records to fleet database first
    # convert_repair_records_to_fleet_database(
    #     Path("./repair_shop_records.csv"),
    #     fleet_db_path
    # )
