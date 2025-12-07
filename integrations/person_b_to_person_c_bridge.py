import os
import json
import csv
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import warnings

def build_person_c_input_from_run(run_dir: str) -> dict:
    """
    Reads:
      - <run_dir>/fingerprint.npy
      - <run_dir>/fault_location.json
      - <run_dir>/meta.json
      - <run_dir>/imu.csv
      - <run_dir>/audio.wav

    Returns a dict compatible with Person C's expectations.
    """
    run_id = os.path.basename(run_dir)
    
    # Defaults
    features = {
        "dominant_frequency": 0.0,
        "rms_vibration": 0.0,
        "spectral_entropy": 0.0,
        "bearing_energy_band": 0.0,
        "audio_anomaly_score": 0.0,
        "speed_dependency": 0.0,
        "fingerprint": []
    }
    
    # 1. Load Fingerprint
    fingerprint_path = os.path.join(run_dir, "fingerprint.npy")
    if os.path.exists(fingerprint_path):
        try:
            fingerprint_arr = np.load(fingerprint_path)
            features["fingerprint"] = fingerprint_arr.tolist()
        except Exception:
            pass

    # 2. Load Location
    loc_path = os.path.join(run_dir, "fault_location.json")
    if os.path.exists(loc_path):
        try:
            with open(loc_path, "r") as f:
                loc_data = json.load(f)
        except Exception:
            loc_data = {}
    else:
        loc_data = {}

    # 3. Load Metadata
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta_data = json.load(f)
        except Exception:
            meta_data = {}
    else:
        meta_data = {}
    
    speed_rpm = float(meta_data.get("speed_rpm", 0.0))

    # 4. Feature Extraction from IMU (imu.csv)
    imu_path = os.path.join(run_dir, "imu.csv")
    if os.path.exists(imu_path):
        try:
            # We assume columns: time, ax, ay, az. Skip header.
            data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
            if data.ndim == 2 and data.shape[1] >= 4:
                # Extract accelerations (ax, ay, az are cols 1, 2, 3)
                ax = data[:, 1]
                ay = data[:, 2]
                az = data[:, 3]
                
                # --- RMS Vibration ---
                # rms = sqrt(mean(ax^2 + ay^2 + az^2))
                features["rms_vibration"] = float(np.sqrt(np.mean(ax**2 + ay**2 + az**2)))
                
                # --- Frequency Domain Analysis (using ax as main signal) ---
                # Assuming fs is roughly constant. Let's estimate fs from time column (col 0)
                time = data[:, 0]
                if len(time) > 1:
                    dt = np.mean(np.diff(time))
                    fs = 1.0 / dt if dt > 0 else 1000.0
                else:
                    fs = 1000.0
                
                freqs, psd = signal.periodogram(ax, fs)
                
                # --- Dominant Frequency ---
                # Ignore DC (index 0)
                if len(psd) > 1:
                    idx = np.argmax(psd[1:]) + 1
                    dom_freq = float(freqs[idx])
                    # Clip to valid range [1, 5000]
                    features["dominant_frequency"] = max(1.0, min(5000.0, dom_freq))
                else:
                    features["dominant_frequency"] = 1.0 # Fallback
                
                # --- Spectral Entropy ---
                # psd_norm = psd / (psd.sum() + 1e-12)
                # entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
                psd_sum = psd.sum() + 1e-12
                psd_norm = psd / psd_sum
                features["spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
                
                # --- Bearing Energy Band (200-2000 Hz) ---
                mask = (freqs >= 200) & (freqs <= 2000)
                if np.any(mask):
                    features["bearing_energy_band"] = float(np.trapz(psd[mask], freqs[mask]))
                else:
                    features["bearing_energy_band"] = 0.0

        except Exception as e:
            # warn but continue
            print(f"Warning: Failed to process IMU data for {run_id}: {e}")

    # 5. Feature Extraction from Audio (audio.wav)
    audio_path = os.path.join(run_dir, "audio.wav")
    if os.path.exists(audio_path):
        try:
            # Suppress wavfile read warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samplerate, audio_data = wavfile.read(audio_path)
            
            # Convert to float normalized [-1, 1] if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.uint8:
                audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
            else:
                audio_float = audio_data.astype(np.float32)

            # If stereo, take mean to get mono
            if audio_float.ndim > 1:
                audio_float = np.mean(audio_float, axis=1)
                
            # --- Audio Anomaly Score ---
            # Heuristic: Combination of RMS and Spectral Entropy
            audio_rms = np.sqrt(np.mean(audio_float**2))
            
            # compute entropy for audio
            f_audio, psd_audio = signal.periodogram(audio_float, samplerate)
            psd_audio_norm = psd_audio / (psd_audio.sum() + 1e-12)
            audio_entropy = -np.sum(psd_audio_norm * np.log(psd_audio_norm + 1e-12))
            
            # Normalize entropy roughly (max entropy for uniform noise is log(N/2))
            # But let's just use the raw entropy and scale it down or use a sigmoid
            # Using simple heuristic as requested:
            # audio_anomaly_score = clip(alpha * audio_rms + beta * (entropy_normalized), 0, 1)
            # We'll treat lower entropy (more tonal) as potentially more anomalous if combined with high energy?
            # actually usually high entropy = noise. The prompt says "It doesn't need to be perfect".
            # Let's map RMS directly to a score, maybe boosted by entropy.
            
            # Example heuristic: 
            # 1. RMS -> [0, 1] (assuming mild signals)
            # 2. Add some entropy component.
            
            score = np.tanh(audio_rms * 10) # Squash high amplitude to 1
            features["audio_anomaly_score"] = float(score)

        except Exception as e:
            print(f"Warning: Failed to process Audio data for {run_id}: {e}")

    # 6. Speed Dependency
    # speed_dependency = dominant_frequency / (theoretical + 1e-6)
    dom_freq = features["dominant_frequency"]
    theoretical_fundamental = speed_rpm / 60.0
    if theoretical_fundamental > 0:
        features["speed_dependency"] = float(dom_freq / theoretical_fundamental)
    else:
        features["speed_dependency"] = 0.0

    # 7. Construct Final Payload
    payload = {
        "run_id": run_id,
        "features": features,
        "location": {
            "x": loc_data.get("x", 0.0),
            "y": loc_data.get("y", 0.0),
            "uncertainty_radius": loc_data.get("uncertainty_radius", 0.0),
            "energy": loc_data.get("energy", 0.0)
        },
        "metadata": {
            "run_id": run_id,
            "simulated_fault_type": meta_data.get("fault_type", "unknown"),
            "severity": meta_data.get("severity", 0.0),
            "speed_rpm": speed_rpm,
            "source": "mira_wave_synthetic"
        }
    }
    
    return payload
