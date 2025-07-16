import networkx as nx
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import sympy as sp
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta

# Simulated site data (user-inputtable; lat/lon, metrics aggregated from world models)
sites = {
    'Plant A - Detroit': {'lat': 42.3314, 'lon': -83.0458, 'efficiency': 85, 'downtime': 5, 'risk_score': 20},
    'Plant B - Chicago': {'lat': 41.8781, 'lon': -87.6298, 'efficiency': 92, 'downtime': 2, 'risk_score': 10},
    'Plant C - Austin': {'lat': 30.2672, 'lon': -97.7431, 'efficiency': 78, 'downtime': 10, 'risk_score': 35},
    'Plant D - Seattle': {'lat': 47.6062, 'lon': -122.3321, 'efficiency': 88, 'downtime': 4, 'risk_score': 15}
}

# New: Initial Setup Wizard Sim (Integrations/Uploads/CV Config)
def initial_setup_wizard(G):
    """
    Simulate user-friendly setup: Sync CMDB (API pull), upload docs, config CV—add assets early.
    """
    print("🔧 Initial Setup Wizard - Hybrid Asset Discovery")
    
    # CMDB Sync Sim (e.g., API call to ServiceNow)
    print("📡 Syncing with CMDB (ServiceNow/ServiceNow)...")
    synced_assets = ['CMDBServer', 'OTController', 'SCADASystem', 'NetworkSwitch']  # From 'API'
    for asset in synced_assets:
        G.add_node(asset, type='synced', latency=random.randint(1, 10), source='cmdb',
                   criticality=random.choice(['high', 'medium', 'low']),
                   last_maintenance=datetime.now() - timedelta(days=random.randint(0, 365)),
                   failure_probability=random.uniform(0.01, 0.3))
        if G.nodes:
            G.add_edge(list(G.nodes)[0], asset)
    print(f"   ✅ Synced {len(synced_assets)} assets from CMDB")
    
    # Upload Doc (P&ID/PDF parsing)
    print("📄 Processing uploaded documents...")
    G = upload_document(G, "dummy P&ID")
    
    # CV Config (Computer Vision for asset detection)
    print("📷 Configuring computer vision...")
    G = cv_asset_detection(G)
    
    print("✅ Setup Complete: Added synced/uploaded/visual assets to world model.")
    return G

# Document Upload and Parsing
def upload_document(G, doc_content):
    """Simulate document parsing (P&ID, PDFs, etc.)"""
    extracted_assets = ['OfflinePump', 'ValveX', 'ManualValve', 'PressureGauge']
    for asset in extracted_assets:
        G.add_node(asset, type='offline', latency=random.randint(1, 10), source='document',
                   criticality=random.choice(['high', 'medium', 'low']),
                   last_maintenance=datetime.now() - timedelta(days=random.randint(0, 365)),
                   failure_probability=random.uniform(0.01, 0.3))
        if G.nodes:
            G.add_edge(list(G.nodes)[0], asset)
    print(f"   ✅ Extracted {len(extracted_assets)} assets from documents")
    return G

# CV Asset Detection
def cv_asset_detection(G, image_data=None):
    """Simulate computer vision for asset detection"""
    # Simulate PyTorch model for CV
    detected_assets = ['CameraSensorY', 'RobotArmZ', 'ConveyorBelt', 'SafetyBarrier']
    for asset in detected_assets:
        G.add_node(asset, type='visual', latency=random.randint(1, 10), source='cv',
                   criticality=random.choice(['high', 'medium', 'low']),
                   last_maintenance=datetime.now() - timedelta(days=random.randint(0, 365)),
                   failure_probability=random.uniform(0.01, 0.3))
        if G.nodes:
            G.add_edge(asset, list(G.nodes)[0])
    print(f"   ✅ Detected {len(detected_assets)} assets via computer vision")
    return G

# Generate Site Map Visualization
def generate_site_map(return_base64=True, drill_down_site=None):
    """Generate global site map with metrics and drill-down capability"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('RevealGrid Global Site Map & Metrics Dashboard', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xlim(-130, -60)
    ax.set_ylim(20, 55)
    
    # Add US outline (simplified)
    us_outline = np.array([
        [-125, 25], [-125, 50], [-65, 50], [-65, 25], [-125, 25]
    ])
    ax.plot(us_outline[:, 0], us_outline[:, 1], 'k-', alpha=0.3, linewidth=1)
    
    # Plot sites with efficiency-based coloring
    lats = [data['lat'] for data in sites.values()]
    lons = [data['lon'] for data in sites.values()]
    effs = [data['efficiency'] for data in sites.values()]
    risks = [data['risk_score'] for data in sites.values()]
    
    # Size based on risk (larger = higher risk)
    sizes = [100 + risk * 5 for risk in risks]
    
    scatter = ax.scatter(lons, lats, c=effs, cmap='RdYlGn', s=sizes, alpha=0.8, edgecolors='black')
    
    # Add site labels with metrics
    for name, data in sites.items():
        ax.annotate(
            f"{name}\nEff: {data['efficiency']}%\nDown: {data['downtime']}h\nRisk: {data['risk_score']}",
            (data['lon'] + 0.5, data['lat']),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
    
    plt.colorbar(scatter, label='Efficiency %', shrink=0.8)
    
    # Add legend for risk levels
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='High Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, label='Medium Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=4, label='Low Risk')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if return_base64:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    plt.close()
    
    # Simulate drill-down: If site selected, print granular placeholder
    if drill_down_site and drill_down_site in sites:
        print(f"\n🔍 Drilling down to {drill_down_site}:")
        print(f"   Location: {sites[drill_down_site]['lat']:.4f}, {sites[drill_down_site]['lon']:.4f}")
        print(f"   Efficiency: {sites[drill_down_site]['efficiency']}%")
        print(f"   Downtime: {sites[drill_down_site]['downtime']} hours")
        print(f"   Risk Score: {sites[drill_down_site]['risk_score']}")
        
        # Call other functions for granular view (e.g., graph for that site)
        graph = create_sample_graph()  # Placeholder; customize per site
        graph = initial_setup_wizard(graph)  # Run setup for hybrid
        print("\n📊 Site Assets (Hybrid):", discover_assets())
        recs, risks = recommend_automation(graph, use_rl_sim=True)
        print(f"\n⚡ Site Recommendations: {len(recs)}")
        for rec in recs:
            print(f"   - {rec['action']}")
        print(f"\n⚠️  Site Risks: {len(risks)}")
        for risk in risks:
            print(f"   - {risk['description']}")

# Enhanced workflow graph with more realistic OT components
def create_sample_graph():
    G = nx.DiGraph()
    nodes = [
        'TemperatureSensor', 'PressureSensor', 'FlowMeter', 'PLC_Controller', 
        'Actuator_Valve', 'Actuator_Pump', 'ManualCheck_Operator', 'BottlingMachine', 
        'QualityControl_Station', 'OptimusBot_Assembly', 'SafetySystem', 'DataLogger'
    ]
    
    edges = [
        ('TemperatureSensor', 'PLC_Controller'), ('PressureSensor', 'PLC_Controller'),
        ('FlowMeter', 'PLC_Controller'), ('PLC_Controller', 'Actuator_Valve'),
        ('PLC_Controller', 'Actuator_Pump'), ('Actuator_Valve', 'ManualCheck_Operator'),
        ('Actuator_Pump', 'ManualCheck_Operator'), ('ManualCheck_Operator', 'BottlingMachine'),
        ('BottlingMachine', 'QualityControl_Station'), ('OptimusBot_Assembly', 'ManualCheck_Operator'),
        ('SafetySystem', 'PLC_Controller'), ('DataLogger', 'PLC_Controller')
    ]
    
    G.add_edges_from(edges)
    
    # Enhanced node attributes
    for node in nodes:
        G.nodes[node]['type'] = 'manual' if 'Manual' in node else ('robot' if 'Bot' in node else 'automated')
        G.nodes[node]['latency'] = random.randint(1, 15)
        G.nodes[node]['criticality'] = random.choice(['high', 'medium', 'low'])
        G.nodes[node]['last_maintenance'] = datetime.now() - timedelta(days=random.randint(0, 365))
        G.nodes[node]['failure_probability'] = random.uniform(0.01, 0.3)
        G.nodes[node]['source'] = 'auto_discovery'
        
    return G

# Enhanced automation recommendations with anomaly detection
def recommend_automation(G, use_rl_sim=False):
    recommendations = []
    risk_insights = []
    anomalies = []
    
    # Anomaly detection using statistical analysis
    latencies = [G.nodes[node]['latency'] for node in G.nodes]
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    
    for node in G.nodes:
        node_data = G.nodes[node]
        
        # Anomaly detection
        if node_data['latency'] > mean_lat + 2 * std_lat:
            anomalies.append({
                'node': node,
                'anomaly_type': 'high_latency',
                'severity': 'high',
                'description': f"Anomaly detected in '{node}': High latency ({node_data['latency']})—potential vulnerability."
            })
        
        # Risk assessment
        if node_data['latency'] > 8:
            risk_insights.append({
                'node': node,
                'risk_type': 'high_latency',
                'severity': 'high',
                'description': f"Node {node} has latency {node_data['latency']} - potential bottleneck"
            })
        
        if node_data.get('failure_probability', 0) > 0.2:
            risk_insights.append({
                'node': node,
                'risk_type': 'high_failure_risk',
                'severity': 'medium',
                'description': f"Node {node} has {node_data.get('failure_probability', 0):.2%} failure probability"
            })
        
        # Automation recommendations (including hybrid sources)
        if (node_data['type'] == 'manual' or node_data['latency'] > 5 or 
            node_data['criticality'] == 'high' or 
            'offline' in node_data.get('type', '') or 
            'visual' in node_data.get('type', '') or 
            'synced' in node_data.get('type', '')):
            
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            
            if predecessors and successors:
                latency_reduction = min(node_data['latency'] * 0.7, 8)
                source_info = f" (from {node_data.get('source', 'auto_discovery')})"
                
                base_rec = {
                    'node': node,
                    'action': f"Automate '{node}'{source_info} by bridging {predecessors[0]} to {successors[0]} with AI relay",
                    'expected_latency_reduction': f"{latency_reduction:.1f}s",
                    'priority': 'high' if node_data['criticality'] == 'high' else 'medium',
                    'type': 'automation_gap',
                    'source': node_data.get('source', 'auto_discovery')
                }
                
                if use_rl_sim:
                    reward = random.uniform(0.7, 1.0)
                    base_rec['rl_reward'] = reward
                    base_rec['action'] += f" (RL-sim reward: {reward:.2f}—empowers operator sovereignty)"
                
                recommendations.append(base_rec)
    
    return recommendations, risk_insights, anomalies

# Enhanced world model with LSTM-like predictions
class SimpleLSTMPredictor:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.history = []
    
    def predict(self, current_value, trend_factor=0.1):
        """Simple LSTM-like prediction using moving average and trend"""
        self.history.append(current_value)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)
        
        if len(self.history) < 2:
            return current_value
        
        # Simple trend calculation
        trend = (self.history[-1] - self.history[0]) / len(self.history)
        prediction = current_value + trend * trend_factor
        
        return max(1, prediction)  # Ensure positive latency

# Enhanced world model simulation
def build_world_model(G, steps=5):
    sim_results = {
        "initial_latency": sum(G.nodes[n]['latency'] for n in G.nodes),
        "initial_efficiency": 1 / sum(G.nodes[n]['latency'] for n in G.nodes),
        "predictions": {},
        "optimization_steps": [],
        "hybrid_sources": {}
    }
    
    # Track hybrid sources
    sources = {}
    for node in G.nodes:
        source = G.nodes[node].get('source', 'auto_discovery')
        sources[source] = sources.get(source, 0) + 1
    sim_results["hybrid_sources"] = sources
    
    t = sp.symbols('t')  # Time for physics sim
    predictor = SimpleLSTMPredictor()
    
    for step in range(steps):
        step_data = {
            "step": step + 1,
            "total_latency": 0,
            "node_updates": {},
            "efficiency": 0,
            "physics_eq": None
        }
        
        # Update node latencies based on optimization
        for node in G.nodes:
            current_latency = G.nodes[node]['latency']
            
            # Apply optimization if node is high-latency
            if current_latency > 5:
                reduction = min(random.randint(1, 3), current_latency - 1)
                G.nodes[node]['latency'] = current_latency - reduction
                step_data["node_updates"][node] = {
                    "old_latency": current_latency,
                    "new_latency": G.nodes[node]['latency'],
                    "reduction": reduction
                }
            
            step_data["total_latency"] += G.nodes[node]['latency']
        
        # Calculate efficiency
        step_data["efficiency"] = 1 / step_data["total_latency"]
        step_data["physics_eq"] = 1 / (step_data["total_latency"] * t)
        
        # LSTM-like prediction for next step
        predicted_latency = predictor.predict(step_data["total_latency"])
        sim_results["predictions"][f"step_{step+1}_predicted"] = predicted_latency
        
        sim_results["optimization_steps"].append(step_data)
    
    return sim_results

# Enhanced visualization with hybrid source indicators
def visualize_graph(G, risk_insights=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Color nodes based on type and source
    node_colors = []
    node_sizes = []
    
    for node in G.nodes:
        node_data = G.nodes[node]
        
        # Color by type
        if node_data['type'] == 'manual':
            color = '#ff9999'  # Red for manual
        elif node_data['type'] == 'robot':
            color = '#66b3ff'  # Blue for robot
        else:
            color = '#99ff99'  # Green for automated
        
        # Adjust color based on source
        source = node_data.get('source', 'auto_discovery')
        if source == 'cmdb':
            color = '#ffcc99'  # Orange for CMDB
        elif source == 'document':
            color = '#cc99ff'  # Purple for document
        elif source == 'cv':
            color = '#99ccff'  # Light blue for CV
        
        # Adjust color based on risk
        if risk_insights:
            for risk in risk_insights:
                if risk['node'] == node and risk['severity'] == 'high':
                    color = '#ff6666'  # Darker red for high risk
                    break
        
        node_colors.append(color)
        node_sizes.append(300 + G.nodes[node]['latency'] * 20)
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            font_size=8,
            font_weight='bold',
            ax=ax)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99ff99', markersize=10, label='Automated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Manual'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66b3ff', markersize=10, label='Robot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffcc99', markersize=10, label='CMDB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc99ff', markersize=10, label='Document'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#99ccff', markersize=10, label='CV')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

# Visualize Process Mapping with different styles
def visualize_process_mapping(G, style='plant_floor', return_base64=True):
    """Enhanced process mapping with multiple visualization styles"""
    if style == '3d_plant_floor':
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D positions for plant floor layout
        pos_3d = {
            'TemperatureSensor': (0, 0, 0),
            'PressureSensor': (1, 0, 1),
            'FlowMeter': (2, 0, 0),
            'PLC_Controller': (3, 0, 1),
            'Actuator_Valve': (4, 0, 0),
            'Actuator_Pump': (5, 0, 1),
            'ManualCheck_Operator': (6, 1, 2),
            'BottlingMachine': (7, 0, 0),
            'QualityControl_Station': (8, 0, 1),
            'OptimusBot_Assembly': (6, -1, -1),
            'SafetySystem': (0, 1, 1),
            'DataLogger': (8, 1, 2),
            'OfflinePump': (2, 1, 1),
            'ValveX': (4, 1, 2),
            'CameraSensorY': (1, -1, 0),
            'RobotArmZ': (8, -1, -1),
            'CMDBServer': (0, 1, 1),
            'OTController': (8, 1, 2)
        }
        
        xs = [pos_3d.get(n, (random.uniform(0,8), random.uniform(-1,1), random.uniform(-1,2)))[0] for n in G.nodes]
        ys = [pos_3d.get(n, (0,0,0))[1] for n in G.nodes]
        zs = [pos_3d.get(n, (0,0,0))[2] for n in G.nodes]
        
        # Color by source
        colors = []
        for node in G.nodes:
            source = G.nodes[node].get('source', 'auto_discovery')
            if source == 'cmdb':
                colors.append('#ffcc99')
            elif source == 'document':
                colors.append('#cc99ff')
            elif source == 'cv':
                colors.append('#99ccff')
            else:
                colors.append('#99ff99')
        
        ax.scatter(xs, ys, zs, c=colors, s=100)
        
        # Draw edges
        for edge in G.edges:
            if edge[0] in pos_3d and edge[1] in pos_3d:
                x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
                y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
                z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
                ax.plot(x, y, z, 'gray', alpha=0.6)
        
        # Add labels
        for i, node in enumerate(G.nodes):
            ax.text(xs[i], ys[i], zs[i], node, fontsize=8)
        
        ax.set_title('Process Mapping: 3D Plant Floor Layout (Multimodal)')
        ax.set_axis_off()
        
    elif style == 'scatterplot':
        fig, ax = plt.subplots(figsize=(12, 8))
        
        node_list = list(G.nodes)
        x = range(len(node_list))
        y = [G.nodes[node]['latency'] for node in node_list]
        sources = [G.nodes[node].get('source', 'auto_discovery') for node in node_list]
        
        # Color by source
        color_map = {
            'auto_discovery': 'blue',
            'cmdb': 'orange', 
            'document': 'purple',
            'cv': 'lightblue'
        }
        colors = [color_map.get(s, 'gray') for s in sources]
        
        ax.scatter(x, y, c=colors, s=100, alpha=0.7)
        
        # Add labels
        for i, node in enumerate(node_list):
            ax.text(x[i] + 0.1, y[i], node, fontsize=8)
        
        # Add source means
        source_means = {}
        for source in set(sources):
            source_y = [y[i] for i in range(len(sources)) if sources[i] == source]
            if source_y:
                source_means[source] = np.mean(source_y)
                ax.axhline(source_means[source], color=color_map.get(source, 'gray'), 
                          linestyle='--', alpha=0.5, label=f'{source} mean')
        
        ax.legend()
        ax.set_title('Process Mapping: Scatterplot (Latency vs. Node with Source Clustering)')
        ax.set_xlabel('Node Order')
        ax.set_ylabel('Latency')
        
    else:  # Default plant floor 2D
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                edge_color='gray', ax=ax, node_size=1000, font_size=8)
        ax.set_title('Process Mapping: Plant Floor Layout (2D)')
        ax.set_axis_off()
    
    if return_base64:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    plt.close()

# Enhanced asset discovery with hybrid sources
def discover_assets():
    asset_categories = {
        'sensors': ['TemperatureSensor', 'PressureSensor', 'FlowMeter', 'LevelSensor', 'VibrationSensor'],
        'controllers': ['PLC_Controller', 'DCS_System', 'RTU_Unit'],
        'actuators': ['Actuator_Valve', 'Actuator_Pump', 'Motor_Controller', 'Servo_Drive'],
        'safety': ['SafetySystem', 'Emergency_Stop', 'Fire_Suppression'],
        'robots': ['OptimusBot_Assembly', 'Collaborative_Robot', 'AGV_System'],
        'manual': ['ManualCheck_Operator', 'QualityControl_Station', 'Maintenance_Station'],
        'cmdb_synced': ['CMDBServer', 'OTController', 'SCADASystem', 'NetworkSwitch'],
        'document_parsed': ['OfflinePump', 'ValveX', 'ManualValve', 'PressureGauge'],
        'cv_detected': ['CameraSensorY', 'RobotArmZ', 'ConveyorBelt', 'SafetyBarrier']
    }
    
    discovered = []
    for category, assets in asset_categories.items():
        discovered.extend(assets)
    
    return discovered, asset_categories

# Enhanced robot use model with action planning
def robot_use_model(G, task='automate_gap', robot='Optimus'):
    recs, risks, anomalies = recommend_automation(G, use_rl_sim=True)
    
    actions = []
    for rec in recs:
        action = {
            'robot': robot,
            'task': task,
            'target_node': rec['node'],
            'action_description': rec['action'],
            'expected_benefit': rec['expected_latency_reduction'],
            'priority': rec['priority'],
            'source': rec['source'],
            'world_model_adaptation': f"Robot adapts via world model to optimize {rec['node']}"
        }
        actions.append(action)
    
    return actions, risks, anomalies

# Encrypted world model export for security
def export_world_model(G, sim_results, robot_actions):
    world_model = {
        'timestamp': datetime.now().isoformat(),
        'graph': {
            'nodes': {node: dict(G.nodes[node]) for node in G.nodes},
            'edges': list(G.edges)
        },
        'simulation_results': sim_results,
        'robot_actions': robot_actions,
        'metadata': {
            'total_nodes': len(G.nodes),
            'total_edges': len(G.edges),
            'automated_nodes': len([n for n in G.nodes if G.nodes[n]['type'] == 'automated']),
            'manual_nodes': len([n for n in G.nodes if G.nodes[n]['type'] == 'manual']),
            'robot_nodes': len([n for n in G.nodes if G.nodes[n]['type'] == 'robot']),
            'hybrid_sources': sim_results.get('hybrid_sources', {})
        }
    }
    
    # Encrypt the export (simulated)
    json_data = json.dumps(world_model, default=str)
    encrypted_export = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
    
    return world_model, encrypted_export

# Global site metrics aggregation
def get_global_metrics():
    """Calculate global metrics across all sites"""
    total_efficiency = sum(site['efficiency'] for site in sites.values())
    avg_efficiency = total_efficiency / len(sites)
    
    total_downtime = sum(site['downtime'] for site in sites.values())
    avg_downtime = total_downtime / len(sites)
    
    total_risk = sum(site['risk_score'] for site in sites.values())
    avg_risk = total_risk / len(sites)
    
    return {
        'total_sites': len(sites),
        'avg_efficiency': avg_efficiency,
        'avg_downtime': avg_downtime,
        'avg_risk_score': avg_risk,
        'high_risk_sites': len([s for s in sites.values() if s['risk_score'] > 25]),
        'low_efficiency_sites': len([s for s in sites.values() if s['efficiency'] < 80])
    }

# Main demo function with integrated setup
def run_demo(drill_down_site='Plant A - Detroit', mapping_style='3d_plant_floor'):
    print("=== RevealGrid MVP Demo with Integrated Setup & Hybrid Discovery ===\n")
    
    # Global site map
    print("🗺️  Global Site Map Generated")
    site_map_viz = generate_site_map()
    print(f"   Site Map Base64 (snippet): {site_map_viz[:100]}...")
    
    # Global metrics
    global_metrics = get_global_metrics()
    print(f"\n📊 Global Metrics:")
    print(f"   Total Sites: {global_metrics['total_sites']}")
    print(f"   Average Efficiency: {global_metrics['avg_efficiency']:.1f}%")
    print(f"   Average Downtime: {global_metrics['avg_downtime']:.1f} hours")
    print(f"   Average Risk Score: {global_metrics['avg_risk_score']:.1f}%")
    print(f"   High Risk Sites: {global_metrics['high_risk_sites']}")
    print(f"   Low Efficiency Sites: {global_metrics['low_efficiency_sites']}")
    
    # Drill-down to specific site
    generate_site_map(drill_down_site=drill_down_site)
    
    # Create graph and run setup wizard
    graph = create_sample_graph()
    graph = initial_setup_wizard(graph)  # Early setup with hybrid discovery
    
    # Process mapping visualization
    print(f"\n📊 Process Mapping Visualization ({mapping_style}):")
    process_viz = visualize_process_mapping(graph, style=mapping_style)
    print(f"   Process Map Base64 (snippet): {process_viz[:100]}...")
    
    # Asset discovery
    discovered_assets, asset_categories = discover_assets()
    print(f"\n🔍 Hybrid Asset Discovery: {len(discovered_assets)} total")
    for category, assets in asset_categories.items():
        print(f"   {category}: {len(assets)} assets")
    
    # Create and analyze graph
    print(f"\n📈 Graph Analysis:")
    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Edges: {len(graph.edges)}")
    
    # Get recommendations and risk insights with anomalies
    recs, risks, anomalies = recommend_automation(graph, use_rl_sim=True)
    print(f"\n⚡ Automation Recommendations: {len(recs)}")
    for rec in recs:
        print(f"   - {rec['action']}")
    
    print(f"\n⚠️  Risk Insights: {len(risks)}")
    for risk in risks:
        print(f"   - {risk['description']} (Severity: {risk['severity']})")
    
    print(f"\n🚨 Anomalies Detected: {len(anomalies)}")
    for anomaly in anomalies:
        print(f"   - {anomaly['description']}")
    
    # World model simulation
    world_sim = build_world_model(graph)
    print(f"\n🌍 World Model Simulation:")
    print(f"   Initial Latency: {world_sim['initial_latency']:.2f}s")
    print(f"   Final Latency: {world_sim['optimization_steps'][-1]['total_latency']:.2f}s")
    print(f"   Efficiency Improvement: {world_sim['optimization_steps'][-1]['efficiency'] / world_sim['initial_efficiency']:.2%}")
    print(f"   Hybrid Sources: {world_sim.get('hybrid_sources', {})}")
    
    # Robot actions
    robot_actions, robot_risks, robot_anomalies = robot_use_model(graph)
    print(f"\n🤖 Robot Actions: {len(robot_actions)}")
    for action in robot_actions:
        print(f"   - {action['action_description']}")
    
    # Export world model
    world_model, encrypted_export = export_world_model(graph, world_sim, robot_actions)
    print(f"\n📦 World Model Exported: {len(world_model['graph']['nodes'])} nodes, {len(world_model['graph']['edges'])} edges")
    print(f"   Encrypted Export (snippet): {encrypted_export[:100]}...")
    
    # Generate visualization
    viz = visualize_graph(graph, risks)
    print(f"\n📊 Graph Visualization Generated: {len(viz)} base64 characters")
    
    return {
        'sites': sites,
        'global_metrics': global_metrics,
        'site_map': site_map_viz,
        'process_map': process_viz,
        'graph': graph,
        'recommendations': recs,
        'risks': risks,
        'anomalies': anomalies,
        'world_sim': world_sim,
        'robot_actions': robot_actions,
        'world_model': world_model,
        'encrypted_export': encrypted_export,
        'visualization': viz
    }

if __name__ == "__main__":
    results = run_demo() 