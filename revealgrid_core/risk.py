"""
Risk analysis module for RevealGrid.

Handles anomaly detection, risk assessment, and vulnerability analysis
for OT asset networks.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple


def analyze_risks(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Analyze risks in the OT network graph.
    
    Args:
        G: NetworkX graph with assets
        
    Returns:
        List of risk insights
    """
    risks = []
    
    # Analyze latency-based risks
    latencies = [G.nodes[node].get('latency', 0) for node in G.nodes]
    mean_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0
    
    for node in G.nodes:
        latency = G.nodes[node].get('latency', 0)
        failure_prob = G.nodes[node].get('failure_prob', 0)
        
        # High latency risk
        if latency > mean_latency + 2 * std_latency:
            risks.append({
                'node': node,
                'type': 'high_latency',
                'severity': 'high',
                'description': f'Node {node} has latency {latency} - potential bottleneck',
                'value': latency
            })
        
        # High failure probability risk
        if failure_prob > 0.25:
            risks.append({
                'node': node,
                'type': 'high_failure_prob',
                'severity': 'medium',
                'description': f'Node {node} has {failure_prob:.1%} failure probability',
                'value': failure_prob
            })
    
    return risks


def detect_anomalies(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Detect anomalies in the OT network.
    
    Args:
        G: NetworkX graph with assets
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    # Statistical anomaly detection
    latencies = [G.nodes[node].get('latency', 0) for node in G.nodes]
    failure_probs = [G.nodes[node].get('failure_prob', 0) for node in G.nodes]
    
    if latencies:
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        for node in G.nodes:
            latency = G.nodes[node].get('latency', 0)
            
            # Anomaly if latency is > 2 standard deviations from mean
            if latency > mean_latency + 2 * std_latency:
                anomalies.append({
                    'node': node,
                    'type': 'latency_anomaly',
                    'severity': 'high',
                    'description': f'Anomaly detected in {node}: High latency ({latency})—potential vulnerability.',
                    'value': latency
                })
    
    if failure_probs:
        mean_failure = np.mean(failure_probs)
        std_failure = np.std(failure_probs)
        
        for node in G.nodes:
            failure_prob = G.nodes[node].get('failure_prob', 0)
            
            # Anomaly if failure probability is > 2 standard deviations from mean
            if failure_prob > mean_failure + 2 * std_failure:
                anomalies.append({
                    'node': node,
                    'type': 'failure_anomaly',
                    'severity': 'medium',
                    'description': f'Anomaly detected in {node}: High failure probability ({failure_prob:.1%})',
                    'value': failure_prob
                })
    
    return anomalies


def calculate_risk_score(G: nx.DiGraph) -> float:
    """
    Calculate overall risk score for the network.
    
    Args:
        G: NetworkX graph with assets
        
    Returns:
        Risk score between 0 and 1
    """
    if not G.nodes:
        return 0.0
    
    # Factors contributing to risk
    total_nodes = len(G.nodes)
    high_latency_nodes = sum(1 for node in G.nodes if G.nodes[node].get('latency', 0) > 8)
    high_failure_nodes = sum(1 for node in G.nodes if G.nodes[node].get('failure_prob', 0) > 0.25)
    
    # Risk calculation
    latency_risk = high_latency_nodes / total_nodes if total_nodes > 0 else 0
    failure_risk = high_failure_nodes / total_nodes if total_nodes > 0 else 0
    
    # Weighted risk score
    risk_score = 0.6 * latency_risk + 0.4 * failure_risk
    
    return min(risk_score, 1.0)


def generate_risk_report(G: nx.DiGraph) -> str:
    """
    Generate human-readable risk report.
    
    Args:
        G: NetworkX graph with assets
        
    Returns:
        Formatted risk report string
    """
    risks = analyze_risks(G)
    anomalies = detect_anomalies(G)
    risk_score = calculate_risk_score(G)
    
    report = "## Risk Analysis Report\n\n"
    
    # Overall risk score
    report += f"**Overall Risk Score:** {risk_score:.1%}\n\n"
    
    # Risk insights
    report += "### Risk Insights\n"
    for risk in risks:
        severity_icon = "🔴" if risk['severity'] == 'high' else "🟡"
        report += f"{severity_icon} {risk['description']} (Severity: {risk['severity']})\n"
    
    # Anomalies
    report += "\n### Anomalies Detected\n"
    if anomalies:
        for anomaly in anomalies:
            severity_icon = "🔴" if anomaly['severity'] == 'high' else "🟡"
            report += f"{severity_icon} {anomaly['description']}\n"
    else:
        report += "✅ No anomalies detected\n"
    
    # Network metrics
    metrics = {
        'total_nodes': len(G.nodes),
        'total_edges': len(G.edges),
        'high_latency_nodes': sum(1 for node in G.nodes if G.nodes[node].get('latency', 0) > 8),
        'high_failure_nodes': sum(1 for node in G.nodes if G.nodes[node].get('failure_prob', 0) > 0.25)
    }
    
    report += f"\n### Network Metrics\n"
    report += f"- Total Assets: {metrics['total_nodes']}\n"
    report += f"- Total Connections: {metrics['total_edges']}\n"
    report += f"- High Latency Assets: {metrics['high_latency_nodes']}\n"
    report += f"- High Failure Risk Assets: {metrics['high_failure_nodes']}\n"
    
    return report 