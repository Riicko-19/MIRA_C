import numpy as np
import scipy.signal as signal
import json
import os

class PhysicsWavefieldAgent:
    """
    Person B: Physics Wavefield Agent
    Purpose: Analyze IMU vibrations physically, perform modal-style analysis, and localize faults.
    """

    def __init__(self):
        self.imu_data = None
        self.fs = 0
        self.t = None
        self.system_matrix = None
        self.green_matrix = None
        self.grid_size = 32

    # --- Preprocessing ---

    def load_raw_imu(self, imu_csv_path):
        """Load imu.csv."""
        try:
            data = np.loadtxt(imu_csv_path, delimiter=",", skiprows=1)
            self.t = data[:, 0]
            self.imu_data = data[:, 1:4] # ax, ay, az
            
            # Estimate fs
            if len(self.t) > 1:
                self.fs = 1.0 / np.mean(np.diff(self.t))
            else:
                self.fs = 1000.0 # Default
            return True
        except Exception as e:
            print(f"Error loading IMU: {e}")
            return False

    def detrend_signal(self, signal_data):
        """Remove DC and slow drift."""
        return signal.detrend(signal_data, axis=0)

    def bandpass_filter(self, signal_data, fs, lowcut, highcut):
        """Apply bandpass filter."""
        if fs <= 0:
            return signal_data

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Clamp / validate
        if low <= 0 or high >= 0.99 or low >= high:
            return signal_data
            
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.filtfilt(b, a, signal_data, axis=0)
        except Exception:
            # Fall back to unfiltered if anything goes wrong
            return signal_data

    def compute_fft_spectrum(self, signal_data, fs):
        """Compute magnitude spectrum."""
        n = len(signal_data)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        mag = np.abs(np.fft.rfft(signal_data, axis=0))
        return freqs, mag

    # --- Modal / Physics ---

    def construct_system_matrix(self):
        """Create a toy stiffness/mass representation for visualization."""
        # Simplified 3x3 matrix representing the 3 axes/DOFs
        self.system_matrix = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0]
        ])
        return self.system_matrix

    def compute_eigenmodes(self):
        """Compute eigenvalues/eigenvectors."""
        if self.system_matrix is None:
            self.construct_system_matrix()
        vals, vecs = np.linalg.eigh(self.system_matrix)
        return vals, vecs

    def project_signal_to_modes(self, signal_data):
        """Project vibration data onto modes."""
        vals, vecs = self.compute_eigenmodes()
        # Project: modes^T * signal^T
        # signal is (N, 3), vecs is (3, 3)
        projected = np.dot(signal_data, vecs)
        return projected

    # --- Inverse Localization ---

    def construct_green_matrix(self):
        """Construct Green's function matrix for inverse problem."""
        # Map 3 sensors to grid_size x grid_size source locations
        # G is (3, grid_size*grid_size)
        # Simple heuristic: distance based attenuation
        
        sensors = np.array([
            [0.2, 0.5], # Sensor 1 (x-axis approx pos)
            [0.5, 0.5], # Sensor 2 (y-axis approx pos)
            [0.8, 0.5]  # Sensor 3 (z-axis approx pos)
        ])
        
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        n_sources = len(grid_points)
        n_sensors = 3
        
        G = np.zeros((n_sensors, n_sources))
        
        for i in range(n_sensors):
            # Distance from source j to sensor i
            dists = np.linalg.norm(grid_points - sensors[i], axis=1)
            # Green's function: 1/r decay (simplified)
            G[i, :] = 1.0 / (dists + 0.1)
            
        self.green_matrix = G
        return G

    def solve_l1_inverse(self, measured_energy):
        """Solve inverse problem using L1-like heuristic."""
        # measured_energy is (3,) vector of RMS energy per axis
        if self.green_matrix is None:
            self.construct_green_matrix()
            
        # Simple back-projection: G.T * d
        # This is more like a matched filter / beamforming
        source_map = np.dot(self.green_matrix.T, measured_energy)
        return source_map

    def solve_l2_inverse(self, measured_energy):
        """Solve using L2 (Least Squares)."""
        # d = G * m -> m = pinv(G) * d
        if self.green_matrix is None:
            self.construct_green_matrix()
            
        source_map = np.dot(np.linalg.pinv(self.green_matrix), measured_energy)
        return source_map

    def compute_energy_distribution(self):
        """Aggregate energy per grid node."""
        if self.imu_data is None: return np.zeros((self.grid_size, self.grid_size))
        
        # 1. Band-limited RMS for localization (50-500 Hz)
        # This focuses on the fault-relevant frequency range
        filtered_imu = self.bandpass_filter(self.imu_data, self.fs, 50.0, 500.0)
        rms = np.sqrt(np.mean(filtered_imu**2, axis=0))
        
        # 2. Solve inverse
        source_vector = self.solve_l1_inverse(rms)
        
        # 3. Reshape to grid
        heatmap = source_vector.reshape((self.grid_size, self.grid_size))
        
        # Normalize
        mn = np.min(heatmap)
        mx = np.max(heatmap)
        
        # Handle flat heatmap
        if mx - mn < 1e-6:
            return np.zeros_like(heatmap)
            
        heatmap = (heatmap - mn) / (mx - mn)
        return heatmap

    # --- Localization Outputs ---

    def generate_fault_heatmap(self):
        """Produce a 2D numpy array representing fault likelihood."""
        return self.compute_energy_distribution()

    def estimate_fault_coordinates(self, heatmap):
        """Convert heatmap into normalized (x, y)."""
        # Find peak
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Map index to [0, 1]
        y_idx, x_idx = idx
        x = x_idx / (self.grid_size - 1)
        # Image coordinates y is down, but usually we want standard cartesian
        # Let's assume standard image coords for now (0,0 top-left)
        y = y_idx / (self.grid_size - 1)
        
        return x, y

    def estimate_uncertainty_radius(self, heatmap):
        """Estimate radius of energy spread."""
        # Threshold at 0.5 max
        binary = heatmap > 0.5
        area = np.sum(binary)
        
        if area == 0:
            return 0.5 # High uncertainty default
            
        # Area = pi * r^2 (in grid units)
        # r_grid = sqrt(area / pi)
        # r_norm = r_grid / grid_size
        r = np.sqrt(area / np.pi) / self.grid_size
        return r

    def run_analysis(self, imu_path):
        """Main entry point for a run."""
        if not self.load_raw_imu(imu_path):
            return None
            
        # Preprocess
        self.imu_data = self.detrend_signal(self.imu_data)
        
        # Modal Analysis (Optional but exposed)
        projected = self.project_signal_to_modes(self.imu_data)
        mode_energies = np.mean(projected**2, axis=0).tolist()
        
        # Generate Heatmap
        heatmap = self.generate_fault_heatmap()
        
        # Estimate Location
        x, y = self.estimate_fault_coordinates(heatmap)
        r = self.estimate_uncertainty_radius(heatmap)
        
        # Calculate total energy
        energy = np.sum(self.imu_data**2)
        
        return {
            "heatmap": heatmap,
            "location": {
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "uncertainty_radius": round(float(r), 2),
                "energy": round(float(energy), 2)
            },
            "mode_energies": mode_energies
        }
