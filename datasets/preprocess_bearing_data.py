"""
CWRU Bearing Dataset - Preprocessing for Person C Evaluation (UPGRADED)

Extracts features from MATLAB .mat files and formats them for Person C pipeline.

UPGRADES:
- Generates 60+ runs by extracting multiple segments per file
- Preserves detailed sub-type labels: bearing_ball, bearing_inner, bearing_outer, normal
- Ensures near-balanced class distribution
- Sub-type labels stored ONLY in ground_truth (never as Person C input)

The CWRU dataset provides vibration signals from bearing tests with various fault types:
- Normal (baseline)
- Ball fault (B007, B014, B021)
- Inner race fault (IR007, IR014, IR021)
- Outer race fault (OR007, OR014, OR021)

This script:
1. Loads raw MATLAB files using scipy.io.loadmat
2. Extracts MULTIPLE time-series segments per file (sliding window)
3. Computes Person C's 6 features from signal processing
4. Maps fault labels to Person C categories + preserves sub-types
5. Generates pseudo fault_location coordinates
6. Saves to data/processed/runs_features.jsonl

Features computed:
- dominant_frequency: Peak frequency in FFT spectrum
- rms_vibration: Root mean square of vibration amplitude
- spectral_entropy: Shannon entropy of power spectrum
- bearing_energy_band: Energy in bearing fault frequency bands (1-5 kHz)
- audio_anomaly_score: Statistical anomaly score vs baseline
- speed_dependency: Encoded as "medium" (all CWRU data at constant load)
"""

import numpy as np
import scipy.io
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any


# Fault type mapping: CWRU -> Person C categories + detailed sub-types
# Person C receives: "Bearing Wear" or "Normal"
# Evaluation uses: bearing_ball, bearing_inner, bearing_outer, normal
FAULT_MAPPING = {
    "Normal": "Normal",
    "B007": "Bearing Wear",  # Ball fault
    "B014": "Bearing Wear",
    "B021": "Bearing Wear",
    "IR007": "Bearing Wear",  # Inner race
    "IR014": "Bearing Wear",
    "IR021": "Bearing Wear",
    "OR007": "Bearing Wear",  # Outer race
    "OR014": "Bearing Wear",
    "OR021": "Bearing Wear",
}

# Detailed sub-type labels (for ground_truth only)
FAULT_SUBTYPE = {
    "Normal": "normal",
    "B007": "bearing_ball",
    "B014": "bearing_ball",
    "B021": "bearing_ball",
    "IR007": "bearing_inner",
    "IR014": "bearing_inner",
    "IR021": "bearing_inner",
    "OR007": "bearing_outer",
    "OR014": "bearing_outer",
    "OR021": "bearing_outer",
}

# Virtual 2D coordinates for fault locations (pseudo-mapped)
# Normalized to [0, 1] range as required by Person C validation
FAULT_LOCATIONS = {
    "Normal": (0.5, 0.5),      # Center (baseline)
    "B007": (0.80, 0.45),      # Ball faults - right side
    "B014": (0.85, 0.50),
    "B021": (0.90, 0.55),
    "IR007": (0.45, 0.80),     # Inner race - top
    "IR014": (0.50, 0.85),
    "IR021": (0.55, 0.90),
    "OR007": (0.45, 0.20),     # Outer race - bottom
    "OR014": (0.50, 0.15),
    "OR021": (0.55, 0.10),
}

# CWRU dataset parameters
SAMPLING_RATE = 48000  # Hz (48 kHz)
SEGMENT_LENGTH = 4096  # Samples per segment
SEGMENTS_PER_FILE = 8  # Number of non-overlapping segments to extract per file
SEGMENT_STRIDE = 8192  # Stride for segment extraction (50% overlap)


def load_mat_file(file_path: Path) -> np.ndarray:
    """
    Load MATLAB .mat file and extract vibration signal.
    
    CWRU .mat files contain multiple keys - we extract drive end (DE) data.
    
    Args:
        file_path: Path to .mat file
        
    Returns:
        1D array of vibration signal
    """
    mat_data = scipy.io.loadmat(file_path)
    
    # CWRU files have keys like 'X###_DE_time' for drive end accelerometer
    # Find the data key (exclude __header__, __version__, __globals__)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    if not data_keys:
        raise ValueError(f"No data found in {file_path.name}")
    
    # Prefer drive end (DE) time series data
    de_keys = [k for k in data_keys if 'DE_time' in k]
    if de_keys:
        signal = mat_data[de_keys[0]].flatten()
    else:
        # Fallback to first data key
        signal = mat_data[data_keys[0]].flatten()
    
    return signal


def compute_fft_features(signal: np.ndarray, fs: int = SAMPLING_RATE) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute FFT and extract frequency domain features.
    
    Args:
        signal: Time-series vibration signal
        fs: Sampling frequency
        
    Returns:
        (dominant_frequency, power_spectrum, frequencies)
    """
    n = len(signal)
    fft_values = fft(signal)
    frequencies = fftfreq(n, d=1/fs)
    
    # Take positive frequencies only
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power_spectrum = np.abs(fft_values[positive_freq_idx]) ** 2
    
    # Normalize power spectrum
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Find dominant frequency
    peak_idx = np.argmax(power_spectrum)
    dominant_frequency = frequencies[peak_idx]
    
    return dominant_frequency, power_spectrum, frequencies


def compute_rms_vibration(signal: np.ndarray) -> float:
    """
    Compute RMS (root mean square) of vibration signal.
    
    Args:
        signal: Time-series vibration signal
        
    Returns:
        RMS value
    """
    return np.sqrt(np.mean(signal ** 2))


def compute_spectral_entropy(power_spectrum: np.ndarray) -> float:
    """
    Compute Shannon entropy of power spectrum.
    
    Higher entropy = more distributed energy (complex fault patterns).
    Lower entropy = concentrated energy (simple periodic faults).
    
    Args:
        power_spectrum: Normalized power spectrum
        
    Returns:
        Spectral entropy (0-1 normalized)
    """
    # Avoid log(0) issues
    power_spectrum = power_spectrum[power_spectrum > 1e-12]
    
    # Shannon entropy
    ent = entropy(power_spectrum, base=2)
    
    # Normalize to 0-1 range (empirical max entropy ~15 for bearing signals)
    return min(ent / 15.0, 1.0)


def compute_bearing_energy_band(power_spectrum: np.ndarray, frequencies: np.ndarray) -> float:
    """
    Compute energy in bearing fault frequency band (1-5 kHz).
    
    Bearing faults typically manifest in 1-5 kHz range for CWRU dataset.
    
    Args:
        power_spectrum: Normalized power spectrum
        frequencies: Frequency bins
        
    Returns:
        Fraction of energy in bearing band
    """
    bearing_band_mask = (frequencies >= 1000) & (frequencies <= 5000)
    bearing_energy = np.sum(power_spectrum[bearing_band_mask])
    
    return bearing_energy


def compute_audio_anomaly_score(signal: np.ndarray, baseline_rms: float = 0.5) -> float:
    """
    Compute anomaly score based on deviation from baseline.
    
    Args:
        signal: Time-series vibration signal
        baseline_rms: Expected RMS for normal condition
        
    Returns:
        Anomaly score (0-1)
    """
    rms = compute_rms_vibration(signal)
    
    # Anomaly score: normalized deviation from baseline
    anomaly = abs(rms - baseline_rms) / (baseline_rms + 1e-6)
    
    # Clip to 0-1 range
    return min(anomaly, 1.0)


def extract_features_from_signal(signal: np.ndarray, baseline_rms: float = 0.5) -> Dict[str, Any]:
    """
    Extract all 6 Person C features from vibration signal.
    
    Args:
        signal: Time-series vibration signal
        baseline_rms: Baseline RMS for anomaly scoring
        
    Returns:
        Dictionary with 6 features
    """
    # FFT features
    dominant_freq, power_spec, frequencies = compute_fft_features(signal)
    
    # Time domain
    rms_vib = compute_rms_vibration(signal)
    
    # Frequency domain
    spec_entropy = compute_spectral_entropy(power_spec)
    bearing_energy = compute_bearing_energy_band(power_spec, frequencies)
    
    # Anomaly detection
    anomaly_score = compute_audio_anomaly_score(signal, baseline_rms)
    
    # Speed dependency: CWRU data at constant load, mark as "medium"
    speed_dep = "medium"
    
    return {
        "dominant_frequency": float(dominant_freq),
        "rms_vibration": float(rms_vib),
        "spectral_entropy": float(spec_entropy),
        "bearing_energy_band": float(bearing_energy),
        "audio_anomaly_score": float(anomaly_score),
        "speed_dependency": speed_dep
    }


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse CWRU filename to extract fault type and severity.
    
    Examples:
        B007_1_123.mat -> ("B007", "Ball fault 0.007 inch")
        IR014_1_175.mat -> ("IR014", "Inner race fault 0.014 inch")
        Time_Normal_1_098.mat -> ("Normal", "Normal baseline")
    
    Args:
        filename: MATLAB filename
        
    Returns:
        (fault_code, description)
    """
    stem = filename.replace(".mat", "")
    
    if "Normal" in stem:
        return "Normal", "Normal baseline"
    
    parts = stem.split("_")
    fault_code = parts[0]  # e.g., "B007", "IR014", "OR021"
    
    # Extract fault type and severity
    if fault_code.startswith("B"):
        fault_type = "Ball fault"
    elif fault_code.startswith("IR"):
        fault_type = "Inner race fault"
    elif fault_code.startswith("OR"):
        fault_type = "Outer race fault"
    else:
        fault_type = "Unknown fault"
    
    severity = fault_code[-3:]  # Last 3 digits: 007, 014, 021
    severity_inch = f"0.{severity} inch"
    
    return fault_code, f"{fault_type} {severity_inch}"


def process_mat_file(file_path: Path, run_id_counter: int, baseline_rms: float, segment_idx: int = 0) -> Dict[str, Any]:
    """
    Process a single segment from a MATLAB file and create Person C diagnostic run.
    
    Args:
        file_path: Path to .mat file
        run_id_counter: Sequential run ID
        baseline_rms: Baseline RMS for anomaly scoring
        segment_idx: Which segment to extract (0-based)
        
    Returns:
        Diagnostic run dictionary
    """
    # Parse filename
    fault_code, description = parse_filename(file_path.name)
    
    # Load signal
    signal = load_mat_file(file_path)
    
    # Extract specific segment with stride
    start_idx = segment_idx * SEGMENT_STRIDE
    end_idx = start_idx + SEGMENT_LENGTH
    
    if end_idx > len(signal):
        # If we've run out of signal, take the last valid segment
        end_idx = len(signal)
        start_idx = max(0, end_idx - SEGMENT_LENGTH)
    
    segment = signal[start_idx:end_idx]
    
    # Skip if segment is too short
    if len(segment) < SEGMENT_LENGTH:
        return None
    
    # Extract features
    features = extract_features_from_signal(segment, baseline_rms)
    
    # Get fault location (normalized to [0, 1] range)
    location_x, location_y = FAULT_LOCATIONS.get(fault_code, (0.5, 0.5))
    
    # Get Person C fault category (coarse)
    person_c_cause = FAULT_MAPPING.get(fault_code, "Unknown")
    
    # Get detailed sub-type (for evaluation only)
    subtype_label = FAULT_SUBTYPE.get(fault_code, "unknown")
    
    # Build diagnostic run
    run = {
        "run_id": f"cwru_{run_id_counter:03d}",
        "fault_location": {
            "x": location_x,
            "y": location_y
        },
        "features": features,
        "metadata": {
            "source": "CWRU Bearing Dataset",
            "original_file": file_path.name,
            "segment_index": segment_idx,
            "fault_code": fault_code,
            "description": description,
            "ground_truth_cause": person_c_cause,  # Coarse label for Person C
            "ground_truth_subtype": subtype_label,  # Fine-grained label for evaluation
            "sampling_rate": SAMPLING_RATE,
            "segment_length": SEGMENT_LENGTH,
            "signal_length_original": len(signal)
        }
    }
    
    return run


def compute_baseline_rms(raw_dir: Path) -> float:
    """
    Compute baseline RMS from normal (healthy) bearing data.
    
    Args:
        raw_dir: Directory containing .mat files
        
    Returns:
        Baseline RMS value
    """
    normal_files = list(raw_dir.glob("*Normal*.mat"))
    
    if not normal_files:
        print("  Warning: No normal baseline file found, using default RMS=0.5")
        return 0.5
    
    print(f"  Computing baseline from: {normal_files[0].name}")
    signal = load_mat_file(normal_files[0])
    
    # Use middle segment
    if len(signal) > SEGMENT_LENGTH:
        start_idx = (len(signal) - SEGMENT_LENGTH) // 2
        signal = signal[start_idx:start_idx + SEGMENT_LENGTH]
    
    baseline_rms = compute_rms_vibration(signal)
    print(f"  Baseline RMS: {baseline_rms:.4f}")
    
    return baseline_rms


def main():
    """Main preprocessing pipeline."""
    print("=" * 70)
    print("CWRU Bearing Dataset - Feature Extraction (UPGRADED)")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "runs_features.jsonl"
    
    print(f"\n1. Input directory: {raw_dir}")
    print(f"2. Output file: {output_file}")
    print(f"3. Target: 60+ runs with balanced classes")
    
    # Find all .mat files
    mat_files = sorted(raw_dir.glob("*.mat"))
    
    if not mat_files:
        print("\n✗ Error: No .mat files found in data/raw/raw/")
        print("   Please run: python datasets/download_kaggle_dataset.py")
        return 1
    
    print(f"\n4. Found {len(mat_files)} MATLAB files")
    
    # Compute baseline RMS from normal data
    print("\n5. Computing baseline statistics...")
    baseline_rms = compute_baseline_rms(raw_dir)
    
    # Process each file - extract multiple segments per file
    print(f"\n6. Extracting {SEGMENTS_PER_FILE} segments per file...")
    runs = []
    run_id_counter = 1
    
    # Group files by sub-type for balanced sampling
    files_by_subtype = {}
    for mat_file in mat_files:
        fault_code, _ = parse_filename(mat_file.name)
        subtype = FAULT_SUBTYPE.get(fault_code, "unknown")
        if subtype not in files_by_subtype:
            files_by_subtype[subtype] = []
        files_by_subtype[subtype].append(mat_file)
    
    print(f"   Files by sub-type:")
    for subtype, files in sorted(files_by_subtype.items()):
        print(f"     {subtype}: {len(files)} file(s)")
    
    # Process all segments from all files
    print(f"\n7. Processing segments...")
    for mat_file in mat_files:
        fault_code, _ = parse_filename(mat_file.name)
        subtype = FAULT_SUBTYPE.get(fault_code, "unknown")
        
        print(f"  {mat_file.name} ({subtype})...")
        
        for seg_idx in range(SEGMENTS_PER_FILE):
            try:
                run = process_mat_file(mat_file, run_id_counter, baseline_rms, seg_idx)
                if run is not None:
                    runs.append(run)
                    run_id_counter += 1
            except Exception as e:
                print(f"    ✗ Error on segment {seg_idx}: {e}")
                continue
    
    # Balance classes by sub-sampling if needed
    print(f"\n8. Balancing class distribution...")
    runs_by_subtype = {}
    for run in runs:
        subtype = run["metadata"]["ground_truth_subtype"]
        if subtype not in runs_by_subtype:
            runs_by_subtype[subtype] = []
        runs_by_subtype[subtype].append(run)
    
    # Find class sizes
    class_sizes = {subtype: len(runs_list) for subtype, runs_list in runs_by_subtype.items()}
    min_class_size = min(class_sizes.values())
    max_class_size = max(class_sizes.values())
    
    # If normal class is significantly smaller, keep all; otherwise balance to 16 per class
    if min_class_size < 10:
        # Keep all normal, subsample others to 2x normal size for 60+ total
        target_per_fault_class = min(20, max_class_size)
        print(f"   Normal class limited to {min_class_size} runs")
        print(f"   Target for fault classes: {target_per_fault_class} runs each")
        
        balanced_runs = []
        for subtype in sorted(runs_by_subtype.keys()):
            runs_list = runs_by_subtype[subtype]
            
            if subtype == "normal":
                # Keep all normal samples
                selected = runs_list
                print(f"     {subtype}: {len(selected)} (kept all)")
            else:
                # Subsample fault classes
                if len(runs_list) > target_per_fault_class:
                    indices = np.linspace(0, len(runs_list) - 1, target_per_fault_class, dtype=int)
                    selected = [runs_list[i] for i in indices]
                    print(f"     {subtype}: {len(runs_list)} → {len(selected)} (subsampled)")
                else:
                    selected = runs_list
                    print(f"     {subtype}: {len(selected)} (kept all)")
            
            balanced_runs.extend(selected)
    else:
        # All classes have enough samples, balance to minimum
        target_per_class = max(15, min_class_size)
        print(f"   Target per class: {target_per_class} runs")
        
        balanced_runs = []
        for subtype in sorted(runs_by_subtype.keys()):
            runs_list = runs_by_subtype[subtype]
            
            if len(runs_list) > target_per_class:
                indices = np.linspace(0, len(runs_list) - 1, target_per_class, dtype=int)
                selected = [runs_list[i] for i in indices]
                print(f"     {subtype}: {len(runs_list)} → {len(selected)} (subsampled)")
            else:
                selected = runs_list
                print(f"     {subtype}: {len(selected)} (kept all)")
            
            balanced_runs.extend(selected)
    
    # Reassign run IDs sequentially
    for idx, run in enumerate(balanced_runs, start=1):
        run["run_id"] = f"cwru_{idx:03d}"
    
    # Save to JSONL
    print(f"\n9. Saving {len(balanced_runs)} runs to {output_file.name}...")
    with open(output_file, 'w') as f:
        for run in balanced_runs:
            f.write(json.dumps(run) + "\n")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    
    # Count by sub-type
    subtype_counts = {}
    for run in balanced_runs:
        subtype = run["metadata"]["ground_truth_subtype"]
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
    
    print(f"\n📊 Distribution of {len(balanced_runs)} runs (by sub-type):")
    for subtype, count in sorted(subtype_counts.items()):
        print(f"   {subtype}: {count} runs")
    
    # Count by coarse label (what Person C sees)
    coarse_counts = {}
    for run in balanced_runs:
        cause = run["metadata"]["ground_truth_cause"]
        coarse_counts[cause] = coarse_counts.get(cause, 0) + 1
    
    print(f"\n📊 Distribution (Person C labels):")
    for cause, count in sorted(coarse_counts.items()):
        print(f"   {cause}: {count} runs")
    
    # Feature statistics
    print(f"\n📈 Feature ranges:")
    all_features = [run["features"] for run in balanced_runs]
    
    feature_names = ["dominant_frequency", "rms_vibration", "spectral_entropy", 
                     "bearing_energy_band", "audio_anomaly_score"]
    
    for feat_name in feature_names:
        values = [f[feat_name] for f in all_features]
        print(f"   {feat_name}: [{min(values):.3f}, {max(values):.3f}]")
    
    print("\n✓ Ready for Person C evaluation!")
    print("  Next: python experiments/run_person_c_on_dataset.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
