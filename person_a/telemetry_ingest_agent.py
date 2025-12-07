import numpy as np
import json
import scipy.signal as signal
import math

class TelemetryIngestAgent:
    """
    Person A: Telemetry Ingest Agent
    Purpose: Convert ANY incoming raw data into clean, normalized IMU + audio signals and structured metadata.
    """

    def __init__(self):
        self.raw_data = None
        self.normalized_imu = None
        self.normalized_audio = None
        self.metadata = {}

    # --- Input Normalization Tools ---

    def normalize_meter_json(self, json_data):
        """Parse 'meter' style JSON into unified numpy arrays."""
        # Example format: {"accel": [{"t": 0, "x": 1, ...}], "audio": [...]}
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        imu_list = data.get("accel", [])
        if imu_list:
            # Extract t, x, y, z
            t = [p.get("t", 0) for p in imu_list]
            x = [p.get("x", 0) for p in imu_list]
            y = [p.get("y", 0) for p in imu_list]
            z = [p.get("z", 0) for p in imu_list]
            self.normalized_imu = np.column_stack((t, x, y, z))
            
        audio_list = data.get("audio", [])
        if audio_list:
            self.normalized_audio = np.array(audio_list)
            
        return True

    def normalize_sensor_packet(self, packet_bytes):
        """Parse binary sensor packet."""
        # Stub for binary parsing
        print("TODO: Implement binary packet parsing")
        return False

    def normalize_image_upload(self, image_data):
        """Stub for image normalization."""
        # TODO: Implement image processing if needed
        pass

    def infer_data_type(self, payload):
        """Decide if payload likely contains vibration, audio, or both."""
        has_imu = False
        has_audio = False
        
        if isinstance(payload, dict):
            if "accel" in payload or "imu" in payload: has_imu = True
            if "audio" in payload or "mic" in payload: has_audio = True
            
        return "imu_audio" if (has_imu and has_audio) else "imu" if has_imu else "audio" if has_audio else "unknown"

    def sanitize_payload(self, payload):
        """Clean input payload."""
        # Remove null bytes, weird chars, etc.
        return payload

    def extract_timestamp(self, payload):
        """Extract base timestamp."""
        return payload.get("timestamp", 0)

    def extract_device_id(self, payload):
        """Extract device ID."""
        return payload.get("device_id", "unknown_device")

    def extract_geo_metadata(self, payload):
        """Extract GPS/Location data."""
        return payload.get("location", {})

    def check_sampling_rate(self, time_array):
        """Inspect time arrays to estimate sample rate."""
        if len(time_array) < 2: return 0
        dt = np.diff(time_array)
        avg_dt = np.mean(dt)
        if avg_dt == 0: return 0
        return 1.0 / avg_dt

    def fix_missing_values(self, signal_array):
        """Clean NaNs, strings, type mismatches."""
        # Assume numpy array
        if np.isnan(signal_array).any():
            # Simple forward fill or zero fill
            mask = np.isnan(signal_array)
            signal_array[mask] = 0.0
        return signal_array

    def convert_to_float(self, data):
        """Ensure data is float type."""
        return np.array(data, dtype=float)

    def sync_audio_imu(self):
        """Time-align audio and IMU if both present."""
        # Stub: assume they start at same time for now
        pass

    def resample_to_standard_rate(self, target_imu_fs=1000, target_audio_fs=16000):
        """Convert signals to standard rates."""
        if self.normalized_imu is not None:
            t = self.normalized_imu[:, 0]
            current_fs = self.check_sampling_rate(t)
            if current_fs > 0 and abs(current_fs - target_imu_fs) > 10:
                # Resample IMU
                num_samples = int(len(t) * target_imu_fs / current_fs)
                new_imu = signal.resample(self.normalized_imu, num_samples)
                # Reconstruct time vector
                new_t = np.linspace(t[0], t[-1], num_samples)
                new_imu[:, 0] = new_t
                self.normalized_imu = new_imu

        if self.normalized_audio is not None:
            # Assume audio doesn't have explicit time vector usually, just FS
            # If we knew original FS, we could resample. 
            # For now, pass.
            pass

    def trim_invalid_edges(self):
        """Remove startup/shutdown transients."""
        # Simple trim of first/last 5%
        if self.normalized_imu is not None:
            n = len(self.normalized_imu)
            start = int(n * 0.05)
            end = int(n * 0.95)
            self.normalized_imu = self.normalized_imu[start:end]
            
        if self.normalized_audio is not None:
            n = len(self.normalized_audio)
            start = int(n * 0.05)
            end = int(n * 0.95)
            self.normalized_audio = self.normalized_audio[start:end]

    def denoise_signal(self):
        """Apply a basic denoising filter."""
        # Lowpass IMU at 200Hz
        if self.normalized_imu is not None:
            b, a = signal.butter(4, 0.4, 'low') # Normalized freq (200/500)
            for i in range(1, 4): # x, y, z
                self.normalized_imu[:, i] = signal.filtfilt(b, a, self.normalized_imu[:, i])

    def segment_into_windows(self, window_size=1000):
        """Segment IMU signal into fixed-length windows."""
        if self.normalized_imu is None:
            return []

        n = len(self.normalized_imu)
        segments = []
        for start in range(0, n, window_size):
            end = start + window_size
            if end <= n:
                segments.append(self.normalized_imu[start:end])
        return segments

    def validate_signal_quality(self):
        """Perform simple checks."""
        if self.normalized_imu is not None:
            if np.std(self.normalized_imu[:, 1:]) < 1e-6:
                return False
        return True

    def write_normalized_output(self, output_dir, run_id, data_manager=None):
        """Save normalized signals and metadata."""
        if data_manager is not None:
            # Use DataManagerAgent
            data_manager.base_dir = output_dir
            data_manager.create_run_folder(run_id)
            
            paths = {}
            if self.normalized_imu is not None:
                # Split into t, ax, ay, az
                t = self.normalized_imu[:, 0]
                ax = self.normalized_imu[:, 1]
                ay = self.normalized_imu[:, 2]
                az = self.normalized_imu[:, 3]
                paths["imu"] = data_manager.save_imu_csv(run_id, t, ax, ay, az)
                
            if self.normalized_audio is not None:
                paths["audio"] = data_manager.save_audio_wav(run_id, self.normalized_audio)
                
            # Update metadata
            self.metadata["run_id"] = run_id
            self.metadata["source"] = "ingest"
            paths["meta"] = data_manager.save_meta_json(run_id, self.metadata)
            
            return paths
        else:
            # Fallback
            return {
                "imu": self.normalized_imu,
                "audio": self.normalized_audio,
                "meta": self.metadata
            }
