import os
import sys
import subprocess
import random
import json

# Import the bridge
try:
    from integrations.person_b_to_person_c_bridge import build_person_c_input_from_run
except ImportError:
    # Fallback if running from a different cwd context, though sys.path should handle it if run from root
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from integrations.person_b_to_person_c_bridge import build_person_c_input_from_run

def run_module(module_name):
    """Runs a python module using subprocess."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {module_name}")
    print(f"{'='*60}\n")
    subprocess.check_call([sys.executable, "-m", module_name])

def print_run_report(run_id, runs_dir="runs"):
    """Prints a mini report for a specific run."""
    run_path = os.path.join(runs_dir, run_id)
    meta_path = os.path.join(run_path, "meta.json")
    loc_path = os.path.join(run_path, "fault_location.json")
    fingerprint_path = os.path.join(run_path, "fingerprint.npy")
    heatmap_path = os.path.join(run_path, "heatmap.png")
    spectrogram_path = os.path.join(run_path, "spectrogram.png")
    person_c_input_path = os.path.join(run_path, "person_c_input.json")
    
    print(f"\nRun: {run_id}")
    
    # Meta
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            print(f"  Fault (from meta.json): {meta.get('fault_type')}, severity={meta.get('severity')}")
    else:
        print("  Meta: Not found")
        
    # Location
    if os.path.exists(loc_path):
        with open(loc_path, "r") as f:
            loc = json.load(f)
            print(f"  Localized at: (x={loc.get('x')}, y={loc.get('y')}), radius={loc.get('uncertainty_radius')}")
            print(f"  Total energy: {loc.get('energy')}")
    else:
        print("  Location: Not found")
        
    # Artifacts
    print(f"  Fingerprint path: {fingerprint_path if os.path.exists(fingerprint_path) else 'MISSING'}")
    print(f"  Heatmap: {heatmap_path if os.path.exists(heatmap_path) else 'MISSING'}")
    print(f"  Spectrogram: {spectrogram_path if os.path.exists(spectrogram_path) else 'MISSING'}")
    print(f"  Person C input JSON: {person_c_input_path if os.path.exists(person_c_input_path) else 'MISSING'}")

def main():
    print("MIRA WAVE: END-TO-END DEMO (Person A -> Person B -> Person C Prep)")
    
    # 1. Run Person A (Generation)
    try:
        run_module("person_a.generate_synthetic_fleet")
    except subprocess.CalledProcessError as e:
        print(f"Error running Person A generation: {e}")
        return

    # 2. Run Person B (Pipeline)
    try:
        run_module("person_b.run_person_b_pipeline")
    except subprocess.CalledProcessError as e:
        print(f"Error running Person B pipeline: {e}")
        return

    # 3. Random Sampling Report & Bridge Generation
    print(f"\n{'='*60}")
    print("DEMO REPORT: Random Sample of 3 Runs")
    print(f"{'='*60}")
    
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        all_runs = [d for d in os.listdir(runs_dir) if d.startswith("run_")]
        if all_runs:
            sample_runs = random.sample(all_runs, min(3, len(all_runs)))
            sample_runs.sort()
            for r in sample_runs:
                run_path = os.path.join(runs_dir, r)
                
                # Generate Person C Input
                payload = build_person_c_input_from_run(run_path)
                person_c_input_path = os.path.join(run_path, "person_c_input.json")
                with open(person_c_input_path, "w") as f:
                    json.dump(payload, f, indent=2)
                
                print_run_report(r, runs_dir)
        else:
            print("No runs found in runs/")
    else:
        print("runs/ directory not found.")

if __name__ == "__main__":
    main()
