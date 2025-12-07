import os
import json
import sys

# Ensure we can import from the integrations package or sibling files
# We add the project root (parent directory of this script) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from integrations.person_b_to_person_c_bridge import build_person_c_input_from_run
except ImportError:
    # Fallback: try sibling import if package structure isn't resolved
    try:
        from person_b_to_person_c_bridge import build_person_c_input_from_run
    except ImportError:
        print("Error: Could not import 'build_person_c_input_from_run'.")
        print("Make sure you are running this script from the project root or the integrations directory.")
        sys.exit(1)

def main():
    runs_dir = "runs"
    
    # Check if runs directory exists
    if not os.path.exists(runs_dir):
        print(f"Directory '{runs_dir}' not found. Please run this script from the project root where 'runs/' exists.")
        return

    # Find all subdirectories starting with "run_"
    try:
        all_entries = os.listdir(runs_dir)
    except OSError as e:
        print(f"Error accessing '{runs_dir}': {e}")
        return

    run_dirs = [d for d in all_entries if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
    
    total_runs = 0
    success_count = 0
    failed_count = 0

    print(f"Found {len(run_dirs)} run directories in '{runs_dir}'. Processing...")

    for run_id in run_dirs:
        total_runs += 1
        run_path = os.path.join(runs_dir, run_id)
        
        try:
            # Build the payload using the existing bridge helper
            payload = build_person_c_input_from_run(run_path)
            
            # Save the payload
            out_path = os.path.join(run_path, "person_c_input.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"Skipping {run_id}: {e}")
            failed_count += 1

    print(f"\nProcessed: {total_runs}")
    print(f"Success: {success_count}")
    print(f"Failed/Skipped: {failed_count}")

if __name__ == "__main__":
    main()
