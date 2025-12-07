"""Quick script to analyze Normal run features."""
import json

# Load data
data = [json.loads(line) for line in open('data/processed/runs_features.jsonl')]

# Filter Normal and Fault runs
normal_runs = [r for r in data if r['metadata'].get('ground_truth_cause') == 'Normal']
fault_runs = [r for r in data if r['metadata'].get('ground_truth_cause') != 'Normal']

print("=" * 70)
print("NORMAL RUN ANALYSIS")
print("=" * 70)
print(f"\nTotal Normal runs: {len(normal_runs)}")
print(f"Total Fault runs: {len(fault_runs)}")

print("\n" + "-" * 70)
print("NORMAL RUN FEATURES:")
print("-" * 70)
for r in normal_runs:
    f = r['features']
    print(f"{r['run_id']}: "
          f"rms={f['rms_vibration']:.3f}, "
          f"bearing={f['bearing_energy_band']:.3f}, "
          f"entropy={f['spectral_entropy']:.3f}, "
          f"anomaly={f['audio_anomaly_score']:.3f}, "
          f"freq={f['dominant_frequency']:.1f}Hz")

print("\n" + "-" * 70)
print("FAULT RUN STATISTICS (for comparison):")
print("-" * 70)
rms_fault = [r['features']['rms_vibration'] for r in fault_runs]
bearing_fault = [r['features']['bearing_energy_band'] for r in fault_runs]
entropy_fault = [r['features']['spectral_entropy'] for r in fault_runs]
anomaly_fault = [r['features']['audio_anomaly_score'] for r in fault_runs]

print(f"RMS Vibration:        min={min(rms_fault):.3f}, max={max(rms_fault):.3f}, mean={sum(rms_fault)/len(rms_fault):.3f}")
print(f"Bearing Energy:       min={min(bearing_fault):.3f}, max={max(bearing_fault):.3f}, mean={sum(bearing_fault)/len(bearing_fault):.3f}")
print(f"Spectral Entropy:     min={min(entropy_fault):.3f}, max={max(entropy_fault):.3f}, mean={sum(entropy_fault)/len(entropy_fault):.3f}")
print(f"Audio Anomaly:        min={min(anomaly_fault):.3f}, max={max(anomaly_fault):.3f}, mean={sum(anomaly_fault)/len(anomaly_fault):.3f}")

print("\n" + "-" * 70)
print("NORMAL RUN STATISTICS:")
print("-" * 70)
rms_normal = [r['features']['rms_vibration'] for r in normal_runs]
bearing_normal = [r['features']['bearing_energy_band'] for r in normal_runs]
entropy_normal = [r['features']['spectral_entropy'] for r in normal_runs]
anomaly_normal = [r['features']['audio_anomaly_score'] for r in normal_runs]

print(f"RMS Vibration:        min={min(rms_normal):.3f}, max={max(rms_normal):.3f}, mean={sum(rms_normal)/len(rms_normal):.3f}")
print(f"Bearing Energy:       min={min(bearing_normal):.3f}, max={max(bearing_normal):.3f}, mean={sum(bearing_normal)/len(bearing_normal):.3f}")
print(f"Spectral Entropy:     min={min(entropy_normal):.3f}, max={max(entropy_normal):.3f}, mean={sum(entropy_normal)/len(entropy_normal):.3f}")
print(f"Audio Anomaly:        min={min(anomaly_normal):.3f}, max={max(anomaly_normal):.3f}, mean={sum(anomaly_normal)/len(anomaly_normal):.3f}")

print("\n" + "=" * 70)
