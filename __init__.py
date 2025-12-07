"""
MIRA Wave Person C: Multi-Agent Diagnostic Reasoning System

A fleet-level causal diagnostic intelligence system for automotive fault analysis.
"""

__version__ = "1.0.0"
__author__ = "MIRA Wave Person C"

from .fleet_matching_agent import FleetMatchingAgent
from .causal_inference_agent import CausalInferenceAgent
from .active_experiment_agent import ActiveExperimentAgent
from .scheduler_agent import SchedulerAgent
from .explanation_agent import ExplanationAgent
from .pipeline_runner import run_person_c_pipeline

__all__ = [
    "FleetMatchingAgent",
    "CausalInferenceAgent",
    "ActiveExperimentAgent",
    "SchedulerAgent",
    "ExplanationAgent",
    "run_person_c_pipeline",
]
