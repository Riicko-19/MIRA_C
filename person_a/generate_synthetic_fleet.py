import sys
import os
import random
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person_a.simulation_engine_agent import SimulationEngineAgent
from person_a.data_manager_agent import DataManagerAgent

def main():
    print("Starting Synthetic Fleet Generation (Person A)...")
    
    # Configuration
    NUM_RUNS = 50
    FAULT_TYPES = ["normal", "imbalance", "loose_mount", "bearing_wear"]
    
    # Instantiate Agents
    sim_agent = SimulationEngineAgent()
    data_manager = DataManagerAgent(base_dir="runs")
    
    # Track stats
    stats = {ft: 0 for ft in FAULT_TYPES}
    
    start_time = time.time()
    
    for i in range(1, NUM_RUNS + 1):
        run_id = data_manager.generate_run_id(i)
        
        # Random parameters
        fault_type = random.choice(FAULT_TYPES)
        severity = round(random.uniform(0.1, 1.0), 2)
        
        # Generate Data
        # Note: generate_run calls apply_fault_forces internally with random speed if not specified,
        # but here we rely on generate_run's internal logic which sets speed.
        # However, SimulationEngineAgent.generate_run() generates its own speed.
        
        run_data = sim_agent.generate_run(
            fault_type=fault_type,
            severity=severity,
            duration_sec=5.0,
            imu_fs=1000.0,
            audio_fs=16000.0
        )
        
        # Create Folder
        data_manager.create_run_folder(run_id)
        
        # Save Files
        data_manager.save_imu_csv(
            run_id, 
            run_data["t"], 
            run_data["imu"][:, 0], 
            run_data["imu"][:, 1], 
            run_data["imu"][:, 2]
        )
        
        data_manager.save_audio_wav(
            run_id, 
            run_data["audio"]
        )
        
        # Add source to meta
        meta = run_data["meta"]
        meta["source"] = "simulation"
        data_manager.save_meta_json(run_id, meta)
        
        stats[fault_type] += 1
        
        if i % 10 == 0:
            print(f"Generated {i}/{NUM_RUNS} runs...")

    # Index Manifest
    manifest_path = data_manager.index_dataset_manifest()
    
    elapsed = time.time() - start_time
    print(f"\nGeneration Complete in {elapsed:.2f}s.")
    print(f"Manifest saved to: {manifest_path}")
    print("Run Summary:")
    for ft, count in stats.items():
        print(f"  - {ft}: {count}")

if __name__ == "__main__":
    main()
