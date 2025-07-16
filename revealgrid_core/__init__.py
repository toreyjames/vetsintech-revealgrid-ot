"""
RevealGrid - Industrial Asset Intelligence Platform

A comprehensive OT asset discovery and intelligence platform with AI-powered
process analysis, risk assessment, and world model simulation.
"""

__version__ = "0.1.0"
__author__ = "RevealGrid Team"
__email__ = "founder@revealgrid.ai"

# Import only functions that actually exist
try:
    from .confidence import calc_confidence, SOURCE_WEIGHTS
except ImportError:
    pass

try:
    from .graph_utils import create_sample_graph, visualize_process_mapping
except ImportError:
    pass

try:
    from .risk import analyze_risks, detect_anomalies
except ImportError:
    pass

try:
    from .cv_detect import cv_asset_detection
except ImportError:
    pass

try:
    from .rl_gap import recommend_automation, robot_use_model
except ImportError:
    pass

try:
    from .standards import classify_asset_per_standard, STANDARDS
except ImportError:
    pass

__all__ = [
    "calc_confidence",
    "SOURCE_WEIGHTS",
    "create_sample_graph",
    "visualize_process_mapping",
    "analyze_risks",
    "detect_anomalies",
    "cv_asset_detection",
    "recommend_automation",
    "robot_use_model",
    "classify_asset_per_standard",
    "STANDARDS",
] 