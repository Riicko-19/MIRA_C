import os
import json
import numpy as np
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person_b.physics_wavefield_agent import PhysicsWavefieldAgent
from person_b.fingerprinting_agent import FingerprintingAgent
from person_b.heatmap_visualization_agent import HeatmapVisualizationAgent

RUNS_DIR = "runs"

def main():
    print(f"Starting Person B Pipeline on {RUNS_DIR}...")
    
    if not os.path.exists(RUNS_DIR):
        print(f"Directory {RUNS_DIR} not found.")
        return

    # Instantiate Agents
    physics_agent = PhysicsWavefieldAgent()
    fingerprint_agent = FingerprintingAgent()
    heatmap_agent = HeatmapVisualizationAgent()
    
    runs = [d for d in os.listdir(RUNS_DIR) if d.startswith("run_")]
    runs.sort()
    
    for run_id in runs:
        run_path = os.path.join(RUNS_DIR, run_id)
        imu_path = os.path.join(run_path, "imu.csv")
        
        if not os.path.exists(imu_path):
            print(f"Skipping {run_id}: No imu.csv")
            continue
            
        print(f"Processing {run_id}...")
        
        # 1. Physics & Localization
        physics_result = physics_agent.run_analysis(imu_path)
        if physics_result:
            # Save fault_location.json
            loc_path = os.path.join(run_path, "fault_location.json")
            with open(loc_path, "w") as f:
                json.dump(physics_result["location"], f, indent=2)
                
            # 2. Heatmap Visualization
            heatmap_path = os.path.join(run_path, "heatmap.png")
            heatmap_agent.generate_visualization(physics_result["heatmap"], heatmap_path)
        
        # 3. Fingerprinting
        fingerprint_agent.process_run(run_path)
        
    print("Person B Pipeline Complete.")

if __name__ == "__main__":
    main()
