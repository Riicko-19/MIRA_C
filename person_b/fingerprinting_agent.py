import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import os

class FingerprintingAgent:
    """
    Person B: Fingerprinting Agent
    Purpose: Convert IMU & audio signals into a compact feature vector (fingerprint).
    """

    def __init__(self):
        self.imu_data = None
        self.audio_data = None
        self.audio_fs = 0
        self.imu_fs = 0

    # --- Spectral Tools ---

    def compute_stft(self, signal_data, fs):
        """STFT on signal."""
        f, t, Zxx = signal.stft(signal_data, fs, nperseg=256)
        return f, t, Zxx

    def compute_mel_spectrogram(self, audio_data, fs):
        """Compute mel-scaled spectrogram (simplified)."""
        # Full Mel filterbank is complex without librosa.
        # We will use a simple log-spectrogram as a proxy.
        f, t, Zxx = self.compute_stft(audio_data, fs)
        mag = np.abs(Zxx)
        return f, t, mag

    def normalize_spectrogram(self, mag):
        """Normalize magnitudes (log compression)."""
        return np.log1p(mag)

    def extract_peak_frequencies(self, signal_data, fs, n_peaks=3):
        """Find dominant frequencies."""
        freqs, mag = signal.periodogram(signal_data, fs)
        # Find peaks
        peaks, _ = signal.find_peaks(mag, distance=10)
        if len(peaks) == 0:
            return np.zeros(n_peaks)
            
        peak_mags = mag[peaks]
        sorted_indices = np.argsort(peak_mags)[::-1]
        top_peaks = peaks[sorted_indices[:n_peaks]]
        top_freqs = freqs[top_peaks]
        
        # Pad if fewer than n_peaks
        if len(top_freqs) < n_peaks:
            top_freqs = np.pad(top_freqs, (0, n_peaks - len(top_freqs)))
            
        return top_freqs

    def extract_harmonic_ratios(self, signal_data, fs):
        """Compute ratios between fundamental and harmonic peaks."""
        peaks = self.extract_peak_frequencies(signal_data, fs, n_peaks=2)
        if peaks[0] > 0:
            return peaks[1] / peaks[0]
        return 0.0

    def compute_spectral_entropy(self, signal_data, fs):
        """Compute spectral entropy of the signal."""
        freqs, psd = signal.periodogram(signal_data, fs)
        # Normalize to probability distribution
        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        return entropy

    # --- Embedding Tools ---

    def flatten_spectrogram(self, mag, target_size=64):
        """Flatten part of the spectrogram into a vector."""
        # Resize or crop to fixed size
        flat = mag.flatten()
        if len(flat) > target_size:
            return flat[:target_size]
        else:
            return np.pad(flat, (0, target_size - len(flat)))

    def apply_pca_embedding(self, vector):
        """Reduce dimensions (Stub)."""
        # In a real agent, this would load a pre-trained PCA model.
        # Here we just take a subset or random projection.
        return vector[:16]

    def apply_umap_embedding(self, vector):
        """UMAP Stub."""
        # TODO: Implement UMAP
        return vector[:2]

    # --- Metadata Tools ---

    def combine_imu_audio_features(self, imu_feats, audio_feats):
        """Build final feature vector."""
        return np.concatenate([imu_feats, audio_feats])

    def save_fingerprint_vector(self, vector, output_path):
        """Save as fingerprint.npy."""
        np.save(output_path, vector)

    def plot_spectrogram_image(self, f, t, mag, output_path):
        """Save spectrogram.png."""
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, mag, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def process_run(self, run_dir):
        """Main processing logic."""
        imu_path = os.path.join(run_dir, "imu.csv")
        audio_path = os.path.join(run_dir, "audio.wav")
        
        # Load IMU
        try:
            imu_arr = np.loadtxt(imu_path, delimiter=",", skiprows=1)
            t = imu_arr[:, 0]
            imu_sig = imu_arr[:, 1] # Use X-axis for main features
            
            if len(t) > 1:
                self.imu_fs = 1.0 / np.mean(np.diff(t))
            else:
                self.imu_fs = 1000.0  # fallback default
        except:
            print(f"Failed to load IMU for {run_dir}")
            return False

        # Load Audio
        try:
            self.audio_fs, self.audio_data = wavfile.read(audio_path)
            
            # Convert stereo to mono if needed
            if self.audio_data.ndim > 1:
                self.audio_data = self.audio_data.mean(axis=1)

            # Normalize to -1..1 if int
            if self.audio_data.dtype == np.int16:
                self.audio_data = self.audio_data / 32768.0
        except:
            print(f"Failed to load Audio for {run_dir}")
            return False

        # 1. IMU Features
        imu_peaks = self.extract_peak_frequencies(imu_sig, self.imu_fs)
        imu_rms = np.sqrt(np.mean(imu_sig**2))
        imu_harmonic = self.extract_harmonic_ratios(imu_sig, self.imu_fs)
        imu_entropy = self.compute_spectral_entropy(imu_sig, self.imu_fs)
        
        # Bearing band energy (200-2000 Hz)
        f_p, psd_p = signal.periodogram(imu_sig, self.imu_fs)
        mask = (f_p >= 200) & (f_p <= 2000)
        imu_bearing_energy = np.trapz(psd_p[mask], f_p[mask]) if np.any(mask) else 0.0
        
        # 2. Audio Features
        f, t_spec, mag = self.compute_mel_spectrogram(self.audio_data, self.audio_fs)
        mag_norm = self.normalize_spectrogram(mag)
        audio_peaks = self.extract_peak_frequencies(self.audio_data, self.audio_fs)
        audio_rms = np.sqrt(np.mean(self.audio_data**2))
        
        # 3. Embedding
        spec_flat = self.flatten_spectrogram(mag_norm)
        embedding = self.apply_pca_embedding(spec_flat)
        
        # 4. Combine
        # Vector: [imu_rms, imu_harmonic, imu_peak1, imu_peak2, imu_spectral_entropy, imu_bearing_band_energy, audio_rms, audio_peak1, ...embedding]
        features = np.array([
            imu_rms, imu_harmonic, imu_peaks[0], imu_peaks[1], 
            imu_entropy, imu_bearing_energy,
            audio_rms, audio_peaks[0]
        ])
        fingerprint = self.combine_imu_audio_features(features, embedding)
        
        # 5. Save Outputs
        self.save_fingerprint_vector(fingerprint, os.path.join(run_dir, "fingerprint.npy"))
        self.plot_spectrogram_image(f, t_spec, mag_norm, os.path.join(run_dir, "spectrogram.png"))
        
        return True
