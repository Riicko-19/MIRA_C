import os
import json
import numpy as np
import scipy.io.wavfile as wavfile

class DataManagerAgent:
    """
    Person A: Data Manager Agent
    Purpose: Organize simulation data files and manage dataset manifests.
    """

    def __init__(self, base_dir="runs"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def create_run_folder(self, run_id):
        """Create a folder: runs/run_001/."""
        path = os.path.join(self.base_dir, run_id)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def generate_run_id(self, index):
        """Provide unique run IDs as strings."""
        return f"run_{index:03d}"

    def save_imu_csv(self, run_id, t, ax, ay, az):
        """Save IMU numpy arrays as CSV."""
        run_dir = os.path.join(self.base_dir, run_id)
        data = np.column_stack((t, ax, ay, az))
        path = os.path.join(run_dir, "imu.csv")
        np.savetxt(path, data, delimiter=",", header="time,ax,ay,az", comments="")
        return path

    def save_audio_wav(self, run_id, audio_data, fs=16000):
        """Save audio numpy arrays as WAV."""
        run_dir = os.path.join(self.base_dir, run_id)
        path = os.path.join(run_dir, "audio.wav")
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
            
        wavfile.write(path, fs, audio_data.astype(np.float32))
        return path

    def save_meta_json(self, run_id, meta_dict):
        """Save meta.json."""
        run_dir = os.path.join(self.base_dir, run_id)
        path = os.path.join(run_dir, "meta.json")
        
        # Ensure run_id is in meta
        meta_dict["run_id"] = run_id
        
        with open(path, "w") as f:
            json.dump(meta_dict, f, indent=2)
        return path

    def index_dataset_manifest(self):
        """Maintain a manifest file listing all runs."""
        manifest_path = os.path.join(self.base_dir, "dataset_manifest.jsonl")
        
        # Scan dir
        runs = []
        for d in os.listdir(self.base_dir):
            if d.startswith("run_"):
                runs.append(d)
        
        runs.sort()
        
        with open(manifest_path, "w") as f:
            for r in runs:
                entry = {"run_id": r, "path": os.path.join(self.base_dir, r)}
                f.write(json.dumps(entry) + "\n")
        
        return manifest_path
