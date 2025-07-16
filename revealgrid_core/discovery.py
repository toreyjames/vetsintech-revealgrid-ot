"""
Discovery module for RevealGrid.

Handles loading and merging data from multiple sources with role inference
and process understanding for OT asset discovery.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from . import confidence

def load_passive(pcap_stats: Path) -> pd.DataFrame:
    """
    Load passive scan data from PCAP statistics.
    
    Args:
        pcap_stats: Path to PCAP statistics CSV file
        
    Returns:
        DataFrame with passive scan data
    """
    df = pd.read_csv(pcap_stats)
    df['src'] = 'passive'
    df['role'] = 'unknown'  # To be inferred
    df['world_inputs'] = 'network traffic patterns, device signatures'
    return df

def load_cmdb(csv_file: Path) -> pd.DataFrame:
    """
    Load CMDB data from CSV file.
    
    Args:
        csv_file: Path to CMDB CSV file
        
    Returns:
        DataFrame with CMDB data
    """
    df = pd.read_csv(csv_file)
    df['src'] = 'cmdb'
    df['role'] = df['type'].apply(lambda x: 'control' if 'controller' in str(x).lower() else 'monitoring')
    df['world_inputs'] = 'asset inventory, configuration data, maintenance records'
    return df

def load_pid(pdf_file: Path) -> pd.DataFrame:
    """
    Load P&ID data from PDF file (placeholder for CV/API parsing).
    
    Args:
        pdf_file: Path to P&ID PDF file
        
    Returns:
        DataFrame with P&ID data
    """
    # Placeholder CV/API parse (expand as needed)
    df = pd.DataFrame({
        "asset": ["valve1", "pipe2", "pump3", "tank4"],
        "type": ["control", "flow", "pump", "storage"],
        "src": ["pid"] * 4
    })
    df['role'] = [
        'regulates flow in distillation',
        'transports crude in preheat',
        'circulates cooling water',
        'stores intermediate product'
    ]
    df['world_inputs'] = [
        'pressure sensor, temp reading, flow rate',
        'flow rate, material type, pressure drop',
        'motor current, flow rate, temperature',
        'level sensor, temperature, pressure'
    ]
    return df

def load_enhanced_devices(csv_file: Path) -> pd.DataFrame:
    """
    Load enhanced device data with rich attributes for contextual awareness.
    
    Args:
        csv_file: Path to enhanced devices CSV file
        
    Returns:
        DataFrame with enhanced device data including RL features
    """
    df = pd.read_csv(csv_file)
    
    # Extract RL state features
    df['rl_state_features'] = df.apply(extract_rl_state_features, axis=1)
    
    # Calculate device health metrics
    df['health_score'] = df.apply(calculate_health_score, axis=1)
    
    # Extract operational context
    df['operational_context'] = df.apply(extract_operational_context, axis=1)
    
    return df

def extract_rl_state_features(row: pd.Series) -> List[float]:
    """
    Extract reinforcement learning state features from device data.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        List of RL state features
    """
    features = []
    
    # Basic operational features
    try:
        uptime = float(row.get('uptime_hours', 0))
        features.append(uptime / 8760.0)  # Normalized uptime
    except (ValueError, TypeError):
        features.append(0.5)  # Default normalized uptime
    
    try:
        response_time = float(row.get('response_time_ms', 100))
        features.append(response_time / 1000.0)  # Normalized response time
    except (ValueError, TypeError):
        features.append(0.1)  # Default normalized response time
    
    try:
        data_quality = float(row.get('data_quality_score', 0.5))
        features.append(data_quality)
    except (ValueError, TypeError):
        features.append(0.5)  # Default data quality
    
    # Risk and confidence features
    try:
        risk_score = float(row.get('risk_score', 0.1))
        features.append(risk_score)
    except (ValueError, TypeError):
        features.append(0.1)  # Default risk score
    
    try:
        confidence_level = float(row.get('confidence_level', 0.5))
        features.append(confidence_level)
    except (ValueError, TypeError):
        features.append(0.5)  # Default confidence level
    
    # Environmental features (simplified)
    temp = extract_temperature(row.get('environmental_conditions', ''))
    features.append(temp / 200.0)  # Normalized temperature
    
    # Power consumption feature
    try:
        power = float(row.get('power_consumption_w', 100))
        features.append(power / 5000.0)  # Normalized power
    except (ValueError, TypeError):
        features.append(0.02)  # Default normalized power
    
    # Communication frequency
    try:
        comm_freq = float(row.get('communication_frequency_hz', 10))
        features.append(comm_freq / 100.0)  # Normalized frequency
    except (ValueError, TypeError):
        features.append(0.1)  # Default normalized frequency
    
    return features

def extract_temperature(env_conditions: str) -> float:
    """Extract temperature from environmental conditions string."""
    try:
        if 'Temperature:' in env_conditions:
            temp_str = env_conditions.split('Temperature:')[1].split(',')[0]
            return float(temp_str.replace('C', ''))
        return 25.0  # Default temperature
    except:
        return 25.0

def calculate_health_score(row: pd.Series) -> float:
    """
    Calculate device health score based on multiple factors.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        Health score between 0 and 1
    """
    score = 0.0
    
    # Uptime contribution (30%)
    try:
        uptime = float(row.get('uptime_hours', 0))
        uptime_score = min(uptime / 8760.0, 1.0)  # Normalize to 1 year
        score += uptime_score * 0.3
    except (ValueError, TypeError):
        score += 0.5 * 0.3  # Default uptime score
    
    # Data quality contribution (25%)
    try:
        data_quality = float(row.get('data_quality_score', 0.5))
        score += data_quality * 0.25
    except (ValueError, TypeError):
        score += 0.5 * 0.25  # Default data quality score
    
    # Maintenance status contribution (20%)
    maintenance_due = row.get('next_maintenance', '')
    if maintenance_due:
        # Simple logic: if maintenance is due soon, reduce score
        score += 0.2  # Placeholder - implement date parsing
    
    # Risk score contribution (15%)
    try:
        risk_score = float(row.get('risk_score', 0.1))
        score += (1.0 - risk_score) * 0.15
    except (ValueError, TypeError):
        score += (1.0 - 0.1) * 0.15  # Default risk score
    
    # Operational status contribution (10%)
    status = str(row.get('operational_status', '')).lower()
    if 'operational' in status:
        score += 0.1
    elif 'maintenance' in status:
        score += 0.05
    
    return min(score, 1.0)

def extract_operational_context(row: pd.Series) -> Dict[str, Any]:
    """
    Extract operational context for deep learning and RL.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        Dictionary with operational context
    """
    context = {
        'location': row.get('location', 'Unknown'),
        'process_function': row.get('process_function', 'Unknown'),
        'criticality': row.get('criticality', 'Medium'),
        'network_segment': row.get('network_segment', 'Unknown'),
        'protocols': row.get('protocols', 'Unknown'),
        'environmental_conditions': row.get('environmental_conditions', 'Unknown'),
        'setpoints': row.get('setpoints', 'None'),
        'historical_anomalies': int(row.get('historical_anomalies', 0)),
        'compliance_standards': row.get('compliance_standards', 'None')
    }
    
    return context

def get_device_rl_actions(device_type: str) -> List[str]:
    """
    Get available RL actions for a device type.
    
    Args:
        device_type: Type of device
        
    Returns:
        List of available actions
    """
    action_maps = {
        'controller': ['start', 'stop', 'adjust', 'tune_pid', 'change_setpoint'],
        'sensor': ['calibrate', 'reset', 'change_threshold'],
        'actuator': ['open', 'close', 'throttle', 'adjust_position'],
        'equipment': ['start', 'stop', 'adjust_speed', 'maintenance_mode'],
        'server': ['backup', 'restart', 'update', 'scale']
    }
    
    return action_maps.get(device_type.lower(), ['status_check'])

def get_device_rl_reward_function(device_type: str, process_function: str) -> str:
    """
    Get RL reward function for a device.
    
    Args:
        device_type: Type of device
        process_function: Function of the device in the process
        
    Returns:
        Reward function name
    """
    if 'control' in process_function.lower():
        return 'control_quality_reward'
    elif 'monitor' in process_function.lower():
        return 'accuracy_reward'
    elif 'flow' in process_function.lower():
        return 'flow_control_reward'
    elif 'transfer' in process_function.lower():
        return 'efficiency_reward'
    else:
        return 'general_reward'

def merge_sources(*dfs) -> pd.DataFrame:
    """
    Merge data from multiple sources with role inference and process understanding.
    
    Args:
        *dfs: Variable number of DataFrames to merge
        
    Returns:
        Merged DataFrame with confidence scores and inferred roles
    """
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["asset"])
    
    # Calculate confidence for each row
    merged["confidence"] = merged.apply(confidence.calc_confidence, axis=1)
    
    # Infer roles based on type and source
    merged["role"] = merged.apply(infer_role, axis=1)
    
    # Get world model inputs based on role
    merged["world_inputs"] = merged.apply(get_world_inputs, axis=1)
    
    return merged

def infer_role(row):
    """
    Infer device role based on type and source.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        Inferred role string
    """
    device_type = str(row.get('type', '')).lower()
    source = str(row.get('src', '')).lower()
    
    # Simple logic; expand with ML
    if 'control' in device_type or 'valve' in device_type:
        return 'regulates process flow'
    elif 'pump' in device_type:
        return 'circulates process fluids'
    elif 'sensor' in device_type:
        return 'monitors process parameters'
    elif 'tank' in device_type or 'storage' in device_type:
        return 'stores process materials'
    elif 'cmdb' in source:
        return 'asset management and control'
    elif 'passive' in source:
        return 'network communication node'
    else:
        return 'process equipment'

def get_world_inputs(row):
    """
    Get required world model inputs based on device role.
    
    Args:
        row: DataFrame row with device information
        
    Returns:
        String describing required inputs for world model
    """
    role = str(row.get('role', '')).lower()
    device_type = str(row.get('type', '')).lower()
    
    if 'flow' in role:
        return 'sensor data, P&ID connections, flow rate measurements'
    elif 'monitor' in role:
        return 'real-time sensor readings, alarm thresholds, historical data'
    elif 'control' in role:
        return 'setpoint data, control algorithms, feedback loops'
    elif 'storage' in role:
        return 'level sensors, temperature readings, pressure data'
    elif 'circulate' in role:
        return 'motor current, flow rate, temperature, vibration data'
    else:
        return 'status logs, dependency graph, operational parameters'

def create_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Create sample data for demonstration.
    
    Returns:
        Dictionary with sample DataFrames for different sources
    """
    # Sample passive scan data
    passive_data = {
        'asset': ['PLC_001', 'HMI_002', 'RTU_003', 'SCADA_004'],
        'type': ['controller', 'interface', 'remote', 'system'],
        'ip': ['192.168.1.10', '192.168.1.20', '192.168.1.30', '192.168.1.40'],
        'mac': ['00:11:22:33:44:55', '00:11:22:33:44:66', '00:11:22:33:44:77', '00:11:22:33:44:88']
    }
    
    # Sample CMDB data
    cmdb_data = {
        'asset': ['PLC_001', 'HMI_002', 'Pump_005', 'Valve_006'],
        'type': ['controller', 'interface', 'pump', 'valve'],
        'location': ['Building A', 'Building A', 'Building B', 'Building B'],
        'manufacturer': ['Siemens', 'Allen-Bradley', 'Grundfos', 'Fisher']
    }
    
    # Sample P&ID data
    pid_data = {
        'asset': ['Pump_005', 'Valve_006', 'Tank_007', 'Sensor_008'],
        'type': ['pump', 'valve', 'tank', 'sensor'],
        'process': ['cooling', 'flow control', 'storage', 'temperature']
    }
    
    return {
        'passive': pd.DataFrame(passive_data),
        'cmdb': pd.DataFrame(cmdb_data),
        'pid': pd.DataFrame(pid_data)
    } 