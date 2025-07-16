"""
Reinforcement learning gap analysis module for RevealGrid.

Handles automation recommendations, gap analysis, and RL-based optimization
for OT asset networks.
"""

import networkx as nx
import random
from typing import Dict, List, Any, Tuple


def recommend_automation(G: nx.DiGraph, use_rl_sim: bool = False) -> List[str]:
    """
    Generate automation recommendations using RL simulation.
    
    Args:
        G: NetworkX graph with assets
        use_rl_sim: Whether to use RL simulation for recommendations
        
    Returns:
        List of automation recommendations
    """
    recommendations = []
    
    for node in G.nodes:
        node_type = G.nodes[node].get('type', '')
        latency = G.nodes[node].get('latency', 0)
        
        # Recommend automation for manual processes or high-latency nodes
        if 'manual' in node_type or latency > 5:
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            
            if predecessors and successors:
                base_rec = f"Automate '{node}' (from auto_discovery) by bridging {predecessors[0]} to {successors[0]} with AI relay"
                
                if use_rl_sim:
                    # Simulate RL reward
                    reward = random.uniform(0.7, 1.0)
                    base_rec += f" (RL-sim reward: {reward:.2f}—empowers operator sovereignty)"
                
                recommendations.append(base_rec)
    
    return recommendations


def robot_use_model(G: nx.DiGraph, task: str = 'automate_gap', robot: str = 'Optimus') -> List[str]:
    """
    Generate robot action recommendations.
    
    Args:
        G: NetworkX graph with assets
        task: Type of task for robot
        robot: Robot name
        
    Returns:
        List of robot action recommendations
    """
    recs = recommend_automation(G, use_rl_sim=True)
    actions = []
    
    for rec in recs:
        if 'Anomaly' not in rec:
            action = f"{robot} executes: {rec} – Adapt via world model."
            actions.append(action)
    
    return actions


def analyze_automation_gaps(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze automation gaps in the network.
    
    Args:
        G: NetworkX graph with assets
        
    Returns:
        Dictionary with gap analysis results
    """
    gaps = []
    total_nodes = len(G.nodes)
    
    for node in G.nodes:
        node_type = G.nodes[node].get('type', '')
        latency = G.nodes[node].get('latency', 0)
        
        # Identify automation opportunities
        if 'manual' in node_type:
            gaps.append({
                'node': node,
                'type': 'manual_process',
                'priority': 'high',
                'description': f'Manual process {node} can be automated'
            })
        
        if latency > 8:
            gaps.append({
                'node': node,
                'type': 'high_latency',
                'priority': 'medium',
                'description': f'High latency node {node} needs optimization'
            })
    
    return {
        'total_gaps': len(gaps),
        'high_priority_gaps': len([g for g in gaps if g['priority'] == 'high']),
        'medium_priority_gaps': len([g for g in gaps if g['priority'] == 'medium']),
        'gaps': gaps,
        'automation_potential': len(gaps) / total_nodes if total_nodes > 0 else 0
    }


def simulate_rl_environment(G: nx.DiGraph, steps: int = 10) -> Dict[str, Any]:
    """
    Simulate reinforcement learning environment for optimization.
    
    Args:
        G: NetworkX graph with assets
        steps: Number of simulation steps
        
    Returns:
        Dictionary with RL simulation results
    """
    initial_latency = sum(G.nodes[node].get('latency', 0) for node in G.nodes)
    final_latency = initial_latency * 0.6  # Simulate improvement
    
    simulation_results = {
        'initial_latency': initial_latency,
        'final_latency': final_latency,
        'efficiency_improvement': (initial_latency / final_latency) * 100 if final_latency > 0 else 0,
        'steps': steps,
        'hybrid_sources': {
            'auto_discovery': len([n for n in G.nodes if 'sensor' in G.nodes[n].get('type', '')]),
            'cmdb': len([n for n in G.nodes if 'controller' in G.nodes[n].get('type', '')]),
            'document': len([n for n in G.nodes if 'equipment' in G.nodes[n].get('type', '')]),
            'cv': len([n for n in G.nodes if 'robot' in G.nodes[n].get('type', '')])
        }
    }
    
    return simulation_results


def build_world_model(G: nx.DiGraph, steps: int = 10) -> Dict[str, Any]:
    """
    Build world model simulation for the network.
    
    Args:
        G: NetworkX graph with assets
        steps: Number of simulation steps
        
    Returns:
        Dictionary with world model simulation results
    """
    return simulate_rl_environment(G, steps) 