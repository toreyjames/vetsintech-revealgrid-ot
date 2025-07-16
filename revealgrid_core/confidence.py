"""
Confidence calculation module for RevealGrid.

Handles confidence scoring for individual assets and overall process understanding.
"""

import functools
import operator
import pandas as pd
from typing import Dict, Any

def prod(iterable):
    """Python 3.7-compatible product function."""
    return functools.reduce(operator.mul, iterable, 1)

# Source weights for confidence calculation
SOURCE_WEIGHTS = {
    "passive": 0.4,  # Network scan data
    "cmdb": 0.3,     # Configuration management database
    "pid": 0.2,      # Piping and instrumentation diagrams
    "cv": 0.1        # Computer vision detection
}

def calc_confidence(row):
    """
    Calculate confidence score for a single asset based on data sources.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        Confidence score between 0 and 1
    """
    seen = str(row.get("src", "")).split("|")
    weights = [SOURCE_WEIGHTS.get(s, 0.1) for s in seen]
    return 1 - prod([1 - x for x in weights if x > 0])

def calc_process_confidence(df: pd.DataFrame) -> float:
    """
    Calculate overall plant process confidence.
    
    Args:
        df: DataFrame with device information and confidence scores
        
    Returns:
        Overall process confidence score between 0 and 1
    """
    if df.empty:
        return 0.0
    
    # Weight by device importance and confidence
    device_weights = {
        'controller': 1.0,
        'sensor': 0.8,
        'actuator': 0.9,
        'pump': 0.7,
        'valve': 0.7,
        'tank': 0.6,
        'interface': 0.5,
        'system': 0.8
    }
    
    weighted_confidences = []
    for _, row in df.iterrows():
        device_type = str(row.get('type', '')).lower()
        weight = device_weights.get(device_type, 0.5)
        confidence = row.get('confidence', 0.0)
        weighted_confidences.append(confidence * weight)
    
    if not weighted_confidences:
        return 0.0
    
    return sum(weighted_confidences) / len(weighted_confidences)

def needs_more_data(process_confidence: float, threshold: float = 0.7) -> bool:
    """
    Check if more data is needed based on process confidence.
    
    Args:
        process_confidence: Current process confidence score
        threshold: Confidence threshold for prompting more data
        
    Returns:
        True if more data is needed, False otherwise
    """
    return process_confidence < threshold

def get_data_recommendations(df: pd.DataFrame, process_confidence: float) -> Dict[str, Any]:
    """
    Get recommendations for additional data sources based on current confidence.
    
    Args:
        df: Current device DataFrame
        process_confidence: Current process confidence score
        
    Returns:
        Dictionary with recommendations for improving confidence
    """
    recommendations = {
        'needs_more_data': needs_more_data(process_confidence),
        'suggested_sources': [],
        'missing_roles': [],
        'confidence_gaps': []
    }
    
    if process_confidence < 0.7:
        recommendations['suggested_sources'].append('CMDB upload for asset inventory')
        recommendations['suggested_sources'].append('P&ID upload for process flow understanding')
    
    if process_confidence < 0.5:
        recommendations['suggested_sources'].append('Computer vision for physical asset detection')
        recommendations['suggested_sources'].append('Network traffic analysis for communication patterns')
    
    # Check for missing critical roles
    device_types = df['type'].value_counts()
    if 'controller' not in device_types.index:
        recommendations['missing_roles'].append('Control systems')
    if 'sensor' not in device_types.index:
        recommendations['missing_roles'].append('Monitoring sensors')
    
    # Identify confidence gaps
    low_confidence_devices = df[df['confidence'] < 0.5]
    if not low_confidence_devices.empty:
        recommendations['confidence_gaps'] = low_confidence_devices['asset'].tolist()
    
    return recommendations

def apply_classifier(df: pd.DataFrame, clf) -> pd.DataFrame:
    """
    Apply machine learning classifier to device data.
    
    Args:
        df: DataFrame with device information
        clf: Trained classifier model
        
    Returns:
        DataFrame with predicted classes
    """
    # Assume features exist; in practice, extract from device data
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df["class"] = clf.predict(df[["feature1", "feature2"]])
    else:
        # Fallback: classify based on type
        df["class"] = df["type"].apply(lambda x: "critical" if "controller" in str(x) else "standard")
    
    return df

def calculate_merge_confidence(original_df: pd.DataFrame, new_df: pd.DataFrame) -> float:
    """
    Calculate confidence improvement after merging new data.
    
    Args:
        original_df: Original device DataFrame
        new_df: New data to be merged
        
    Returns:
        Confidence improvement score
    """
    original_conf = calc_process_confidence(original_df)
    
    # Simulate merged DataFrame
    merged_df = pd.concat([original_df, new_df], ignore_index=True)
    merged_conf = calc_process_confidence(merged_df)
    
    return merged_conf - original_conf 