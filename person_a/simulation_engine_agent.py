import numpy as np
from scipy.integrate import odeint
import scipy.signal as signal
import random

class SimulationEngineAgent:
    """
    Person A: Simulation Engine Agent
    Purpose: Generate synthetic vibration, audio, and fault data using a simplified mass-spring-damper vehicle model.
    """

    def __init__(self):
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.damping_matrix = None
        self.state = None # [x, v]
        self.t = None
        self.force_history = None
        self.response_history = None
        
        # Default vehicle parameters (simplified 3-DOF system)
        self.m = [100.0, 50.0, 10.0] # kg (chassis, engine, wheel)
        self.k = [10000.0, 50000.0, 20000.0] # N/m
        self.c = [500.0, 1000.0, 200.0] # Ns/m

        # Safe defaults
        self.current_fault = "normal"
        self.current_severity = 0.0
        self.current_speed_rpm = 1500.0

    # --- Vehicle Physics Tools ---

    def build_mass_matrix(self):
        """Builds the mass matrix for the multi-DOF system."""
        self.mass_matrix = np.diag(self.m)
        return self.mass_matrix

    def build_stiffness_matrix(self):
        """Builds the stiffness matrix."""
        # Simplified coupled system
        k1, k2, k3 = self.k
        # Example coupling: 
        # M1 (chassis) <-> M2 (engine) <-> M3 (wheel)
        self.stiffness_matrix = np.array([
            [k1 + k2, -k2, 0],
            [-k2, k2 + k3, -k3],
            [0, -k3, k3]
        ])
        return self.stiffness_matrix

    def build_damping_matrix(self):
        """Builds the damping matrix."""
        c1, c2, c3 = self.c
        self.damping_matrix = np.array([
            [c1 + c2, -c2, 0],
            [-c2, c2 + c3, -c3],
            [0, -c3, c3]
        ])
        return self.damping_matrix

    def integrate_system_ode(self, t, initial_state=None):
        """Integrates the system equations of motion."""
        if initial_state is None:
            initial_state = np.zeros(2 * len(self.m)) # [x1, x2, x3, v1, v2, v3]
        
        self.t = t
        
        # Ensure matrices are built
        if self.mass_matrix is None: self.build_mass_matrix()
        if self.stiffness_matrix is None: self.build_stiffness_matrix()
        if self.damping_matrix is None: self.build_damping_matrix()
        
        M_inv = np.linalg.inv(self.mass_matrix)
        K = self.stiffness_matrix
        C = self.damping_matrix
        
        def system_dynamics(state, t):
            # state = [x1, x2, x3, v1, v2, v3]
            n = len(self.m)
            x = state[:n]
            v = state[n:]
            
            # Get external force at time t (interpolated or calculated)
            F_ext = self._get_force_at_t(t)
            
            # a = M_inv * (F_ext - C*v - K*x)
            a = M_inv @ (F_ext - C @ v - K @ x)
            
            dxdt = v
            dvdt = a
            return np.concatenate([dxdt, dvdt])

        self.response_history = odeint(system_dynamics, initial_state, t)
        return self.response_history

    def apply_fault_forces(self, fault_type, severity, speed_rpm):
        """Configures the force generation logic based on fault type."""
        self.current_fault = fault_type
        self.current_severity = severity
        self.current_speed_rpm = speed_rpm
        # Force function will be called during integration
        
    def apply_speed_profile(self, t):
        """Generates a speed profile (RPM) over time."""
        # For simplicity, constant speed or slight ramp
        return np.full_like(t, self.current_speed_rpm)

    # --- Fault Tools ---

    def inject_imbalance_force(self, t, speed_rpm, severity):
        """Add a periodic force at a dominant frequency depending on speed."""
        freq_hz = speed_rpm / 60.0
        omega = 2 * np.pi * freq_hz
        force_amp = severity * 1000.0 # N
        
        # Centrifugal force F = m * r * w^2, simplified here
        force = force_amp * np.sin(omega * t)
        
        # Apply mainly to engine mass (M2)
        F = np.zeros((len(t), 3))
        F[:, 1] = force
        return F

    def inject_loose_mount(self, t, severity):
        """Modify stiffness/damping at one DOF to create large low-frequency motion."""
        # In a linear ODE solver, changing K/C mid-integration is hard without callbacks.
        # Here we simulate the EFFECT by adding a low-frequency modulating force 
        # or by returning a force that destabilizes the system.
        
        # Sway frequency
        sway_freq = 2.0 # Hz
        force_amp = severity * 500.0
        
        sway = force_amp * np.sin(2 * np.pi * sway_freq * t)
        
        # Apply to chassis (M1) and engine (M2) in opposition
        F = np.zeros((len(t), 3))
        F[:, 0] = sway
        F[:, 1] = -sway
        return F

    def inject_bearing_noise(self, t, severity):
        """Add broadband high-frequency noise plus discrete impact bursts to the system response."""
        # Broadband noise
        noise = np.random.normal(0, severity * 200.0, len(t))
        
        impact_freq = 10.0  # Hz
        impact_indices = (t * impact_freq).astype(int)
        unique_indices = np.unique(impact_indices)
        
        F = np.zeros((len(t), 3))
        F[:, 2] = noise  # Base noise on wheel/bearing DOF
        
        # Add impulses at selected indices
        for idx in unique_indices:
            if 0 <= idx < len(F):
                F[idx, 2] += severity * 1000.0
                
        return F

    def vary_severity_curve(self, severity_scalar):
        """Map a severity scalar [0, 1] to amplitude/energy of faults."""
        # Non-linear mapping could go here
        return severity_scalar ** 2

    # --- Sensor Tools ---

    def simulate_imu_signal(self, duration_sec=5.0, fs=1000.0):
        """Return a numpy array shape [N, 3] (3-axis acceleration)."""
        t = np.linspace(0, duration_sec, int(duration_sec * fs))
        # Note: linspace includes both endpoints, so dt is slightly less than 1/fs.
        # This is acceptable for our synthetic simulation.
        
        # 1. Setup Physics
        self.build_mass_matrix()
        self.build_stiffness_matrix()
        self.build_damping_matrix()
        
        # 2. Calculate Forces based on current fault config
        F_total = np.zeros((len(t), 3))
        
        if self.current_fault == "imbalance":
            F_total += self.inject_imbalance_force(t, self.current_speed_rpm, self.current_severity)
        elif self.current_fault == "loose_mount":
            F_total += self.inject_loose_mount(t, self.current_severity)
            # Add some base vibration
            F_total += self.inject_imbalance_force(t, self.current_speed_rpm, 0.1)
        elif self.current_fault == "bearing_wear":
            F_total += self.inject_bearing_noise(t, self.current_severity)
            F_total += self.inject_imbalance_force(t, self.current_speed_rpm, 0.1)
        else:
            # Healthy baseline vibration (very low amplitude)
            F_total += self.inject_imbalance_force(t, self.current_speed_rpm, 0.05)
            
        # Store force for ODE (simplified: pre-calculate)
        self._force_cache = (t, F_total)
        
        # 3. Integrate
        response = self.integrate_system_ode(t)
        
        # Response is [x1, x2, x3, v1, v2, v3]
        # We want acceleration. a = M_inv * (F - Cv - Kx)
        # Re-calculate acceleration from state
        accel = np.zeros((len(t), 3))
        M_inv = np.linalg.inv(self.mass_matrix)
        
        for i in range(len(t)):
            x = response[i, :3]
            v = response[i, 3:]
            f = F_total[i]
            a = M_inv @ (f - self.damping_matrix @ v - self.stiffness_matrix @ x)
            accel[i] = a

        # Map 3 DOFs to 3 IMU axes (approximate)
        # x-axis: Chassis fore-aft (M1)
        # y-axis: Engine lateral (M2)
        # z-axis: Wheel vertical (M3)
        return t, accel

    def simulate_audio_signal(self, duration_sec=5.0, fs=16000.0):
        """Return a mono waveform (1D numpy array) that is correlated with vibration."""
        # Audio is often high-frequency vibration coupled to air
        # We can upsample the vibration or generate new correlated noise
        
        t_audio = np.linspace(0, duration_sec, int(duration_sec * fs))
        
        # Re-generate base features at audio rate
        if self.current_fault == "imbalance":
            freq = self.current_speed_rpm / 60.0
            audio = 0.5 * self.current_severity * np.sin(2 * np.pi * freq * t_audio)
            audio += np.random.normal(0, 0.01, len(t_audio))
            
        elif self.current_fault == "loose_mount":
            # Rattle sound
            carrier = np.random.normal(0, 0.1, len(t_audio))
            modulator = np.sin(2 * np.pi * 5.0 * t_audio) # 5Hz rattle
            audio = self.current_severity * carrier * (modulator > 0)
            
        elif self.current_fault == "bearing_wear":
            # High pitched whine
            res_freq = 1000.0
            audio = 0.3 * self.current_severity * np.sin(2 * np.pi * res_freq * t_audio)
            audio += np.random.normal(0, 0.1 * self.current_severity, len(t_audio))
        else:
            audio = np.random.normal(0, 0.01, len(t_audio))
            
        return audio

    def _get_force_at_t(self, t_val):
        """Helper to retrieve force from cache during integration.
        
        If no force cache exists, returns zero external force.
        """
        if not hasattr(self, "_force_cache"):
            # 3 DOFs -> 3 force components
            return np.zeros(3)

        t_arr, F_arr = self._force_cache
        idx = np.searchsorted(t_arr, t_val)
        if idx >= len(t_arr): idx = len(t_arr) - 1
        return F_arr[idx]

    def generate_run(self, fault_type, severity, duration_sec=5.0, imu_fs=1000.0, audio_fs=16000.0):
        """High-level method to generate a full run."""
        speed_rpm = random.uniform(1000, 3000)
        self.apply_fault_forces(fault_type, severity, speed_rpm)
        
        t, imu_data = self.simulate_imu_signal(duration_sec, imu_fs)
        audio_data = self.simulate_audio_signal(duration_sec, audio_fs)
        
        return {
            "imu": imu_data,
            "audio": audio_data,
            "t": t,
            "meta": {
                "fault_type": fault_type,
                "severity": severity,
                "speed_rpm": speed_rpm,
                "duration": duration_sec
            }
        }
