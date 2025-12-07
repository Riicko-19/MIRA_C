"""Quick test for PHASE B upgrades."""

from causal_inference_agent import CausalInferenceAgent
from data_models import DiagnosticRun, FeatureVector, FaultLocation

# Test Stage-0 Normal detection
agent = CausalInferenceAgent()

# Test case 1: Normal features
normal_run = DiagnosticRun(
    run_id="test_normal",
    fault_location=FaultLocation(x=0.5, y=0.5),
    features=FeatureVector(
        dominant_frequency=400.0,
        rms_vibration=0.07,
        spectral_entropy=0.35,
        bearing_energy_band=0.48,
        audio_anomaly_score=0.05,
        speed_dependency="medium"
    )
)

# Test case 2: Bearing fault features
fault_run = DiagnosticRun(
    run_id="test_fault",
    fault_location=FaultLocation(x=0.8, y=0.5),
    features=FeatureVector(
        dominant_frequency=3500.0,
        rms_vibration=0.95,
        spectral_entropy=0.92,
        bearing_energy_band=0.88,
        audio_anomaly_score=0.95,
        speed_dependency="weak"
    )
)

print("=" * 70)
print("PHASE B - Stage-0 Normal Detection Test")
print("=" * 70)

print("\nTest 1: Normal Case")
is_normal, conf, reason = agent.detect_normal_vs_fault(normal_run)
print(f"  is_normal: {is_normal}")
print(f"  confidence: {conf:.3f}")
print(f"  reasoning: {reason}")

print("\nTest 2: Fault Case")
is_normal, conf, reason = agent.detect_normal_vs_fault(fault_run)
print(f"  is_normal: {is_normal}")
print(f"  confidence: {conf:.3f}")
print(f"  reasoning: {reason}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)

# Check that Normal is in fault_signatures
print(f"\nFault signatures include: {list(agent.fault_signatures.keys())}")
print(f"Prior probabilities: {agent.prior_probabilities}")
