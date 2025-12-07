import sys
import os
import json
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person_b.physics_wavefield_agent import PhysicsWavefieldAgent
from person_b.fingerprinting_agent import FingerprintingAgent
from person_b.heatmap_visualization_agent import HeatmapVisualizationAgent

def main():
    runs_dir = "runs"
    print(f"Starting Person B Pipeline on {runs_dir}...")
    
    if not os.path.exists(runs_dir):
        print(f"Directory {runs_dir} not found.")
        return

    # Instantiate Agents
    physics_agent = PhysicsWavefieldAgent()
    fingerprint_agent = FingerprintingAgent()
    heatmap_agent = HeatmapVisualizationAgent()
    
    # Get list of runs
    runs = [d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
    runs.sort()
    
    total_runs = len(runs)
    failed_runs = 0
    
    for idx, run_id in enumerate(runs):
        run_path = os.path.join(runs_dir, run_id)
        imu_path = os.path.join(run_path, "imu.csv")
        audio_path = os.path.join(run_path, "audio.wav")
        meta_path = os.path.join(run_path, "meta.json")
        
        # Check files
        if not (os.path.exists(imu_path) and os.path.exists(audio_path)):
            print(f"[{idx+1}/{total_runs}] Skipping {run_id}: Missing IMU or Audio")
            failed_runs += 1
            continue
            
        # Read meta for logging
        fault_type = "unknown"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    fault_type = meta.get("fault_type", "unknown")
            except:
                pass
        
        print(f"[{idx+1}/{total_runs}] Processing {run_id} (fault_type={fault_type})")
        
        try:
            # 1. Physics & Localization
            physics_result = physics_agent.run_analysis(imu_path)
            if physics_result:
                # Save fault_location.json
                loc_path = os.path.join(run_path, "fault_location.json")
                with open(loc_path, "w") as f:
                    json.dump(physics_result["location"], f, indent=2)
                    
                # Generate Heatmap
                heatmap_path = os.path.join(run_path, "heatmap.png")
                heatmap_agent.generate_visualization(physics_result["heatmap"], heatmap_path)
            else:
                print(f"  Warning: Physics analysis failed for {run_id}")
            
            # 2. Fingerprinting
            success = fingerprint_agent.process_run(run_path)
            if not success:
                print(f"  Warning: Fingerprinting failed for {run_id}")
                
        except Exception as e:
            print(f"  Error processing {run_id}: {e}")
            failed_runs += 1

    print("\nPipeline Complete.")
    print(f"Total Runs Processed: {total_runs}")
    print(f"Failed Runs: {failed_runs}")

if __name__ == "__main__":
    main()
