"""
Graph utilities for RevealGrid.

Handles graph creation, visualization, and process mapping.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection

from .standards import classify_asset_per_standard


def create_sample_graph() -> nx.Graph:
    """Create a sample graph for demonstration."""
    G = nx.Graph()
    
    # Add nodes
    nodes = [
        ('PLC_001', {'type': 'controller', 'location': 'Building A'}),
        ('HMI_002', {'type': 'interface', 'location': 'Building A'}),
        ('RTU_003', {'type': 'remote', 'location': 'Building B'}),
        ('SCADA_004', {'type': 'system', 'location': 'Control Room'}),
        ('Pump_005', {'type': 'pump', 'location': 'Building B'}),
        ('Valve_006', {'type': 'valve', 'location': 'Building B'}),
        ('Tank_007', {'type': 'tank', 'location': 'Building A'}),
        ('Sensor_008', {'type': 'sensor', 'location': 'Building A'})
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('PLC_001', 'HMI_002'),
        ('PLC_001', 'Pump_005'),
        ('PLC_001', 'Valve_006'),
        ('SCADA_004', 'PLC_001'),
        ('RTU_003', 'Tank_007'),
        ('Sensor_008', 'PLC_001')
    ]
    
    G.add_edges_from(edges)
    return G

def create_site_map(device_df) -> str:
    """
    Create a Toyota automotive manufacturing site map visualization.
    
    Args:
        device_df: DataFrame with Toyota manufacturing device information
        
    Returns:
        Base64 encoded image string
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        
        # Toyota manufacturing plant layouts
        plants = {
            'Plant_A': {'x': 15, 'y': 60, 'width': 35, 'height': 35, 'color': '#e6f3ff', 'name': 'Toyota Assembly Plant'},
            'Plant_B': {'x': 55, 'y': 60, 'width': 35, 'height': 35, 'color': '#fff2e6', 'name': 'Toyota Engine Plant'},
            'Data_Center': {'x': 35, 'y': 10, 'width': 30, 'height': 20, 'color': '#f0f8f0', 'name': 'Toyota Control Center'}
        }
        
        # Draw Toyota plant boundaries
        for plant_name, layout in plants.items():
            rect = Rectangle((layout['x'], layout['y']), layout['width'], layout['height'], 
                            linewidth=2, edgecolor='black', facecolor=layout['color'], alpha=0.7)
            ax.add_patch(rect)
            ax.text(layout['x'] + layout['width']/2, layout['y'] + layout['height'] + 2, 
                    layout['name'], ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Toyota automotive equipment colors and symbols
        device_styles = {
            'controller': {'color': '#ff6b6b', 'marker': 's', 'size': 120, 'label': 'PLC Controllers'},
            'robot': {'color': '#4ecdc4', 'marker': '^', 'size': 140, 'label': 'Robots (Fanuc)'},
            'sensor': {'color': '#45b7d1', 'marker': 'o', 'size': 100, 'label': 'Sensors (Keyence)'},
            'equipment': {'color': '#96ceb4', 'marker': 'D', 'size': 110, 'label': 'Production Equipment'},
            'system': {'color': '#feca57', 'marker': 's', 'size': 130, 'label': 'SCADA Systems'},
            'interface': {'color': '#ff9ff3', 'marker': 'h', 'size': 90, 'label': 'HMI Interfaces'},
            'server': {'color': '#54a0ff', 'marker': 's', 'size': 150, 'label': 'Data Servers'}
        }
        
        # Plot Toyota manufacturing devices
        device_positions = {}
        for idx, device in device_df.iterrows():
            device_type = str(device.get('type', 'unknown')).lower()
            physical_loc = str(device.get('physical_location', 'Plant_A'))
            process_loc = str(device.get('process_location', 'Unknown'))
            hostname = str(device.get('hostname', f'Device_{idx}'))
            manufacturer = str(device.get('manufacturer', 'Unknown'))
            
            # Get plant layout
            plant_layout = plants.get(physical_loc, plants['Plant_A'])
            
            # Calculate position within Toyota plant based on process
            if physical_loc == 'Plant_A':
                if 'Assembly' in process_loc:
                    x = plant_layout['x'] + 8
                    y = plant_layout['y'] + 25
                elif 'Paint' in process_loc:
                    x = plant_layout['x'] + 20
                    y = plant_layout['y'] + 15
                elif 'Control' in process_loc:
                    x = plant_layout['x'] + 28
                    y = plant_layout['y'] + 30
                else:
                    x = plant_layout['x'] + 15
                    y = plant_layout['y'] + 10
            elif physical_loc == 'Plant_B':
                if 'Engine' in process_loc:
                    x = plant_layout['x'] + 12
                    y = plant_layout['y'] + 22
                elif 'Test' in process_loc:
                    x = plant_layout['x'] + 25
                    y = plant_layout['y'] + 12
                else:
                    x = plant_layout['x'] + 18
                    y = plant_layout['y'] + 18
            else:  # Data Center
                x = plant_layout['x'] + 15
                y = plant_layout['y'] + 10
            
            # Get device style
            style = device_styles.get(device_type, {'color': 'gray', 'marker': 'o', 'size': 100, 'label': 'Other'})
            
            # Plot device with Toyota branding
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size'], 
                      edgecolors='black', linewidth=1, zorder=5)
            
            # Add device label with manufacturer info
            label_text = f"{hostname}\n({manufacturer})" if manufacturer != 'Unknown' else hostname
            ax.text(x, y - 4, label_text, ha='center', va='top', fontsize=7, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            device_positions[hostname] = (x, y)
        
        # Add Toyota-specific legend
        legend_elements = []
        for device_type, style in device_styles.items():
            legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], color='w', 
                                            markerfacecolor=style['color'], markersize=10, 
                                            label=style['label']))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Add Toyota manufacturing title and labels
        ax.set_title('Toyota Automotive Manufacturing Site Map', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Plant Layout', fontsize=14)
        ax.set_ylabel('Production Areas', fontsize=14)
        
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        
        # Add Toyota manufacturing zones
        zone_labels = [
            {'text': 'Assembly Line 1', 'x': 32, 'y': 75, 'color': '#ff6b6b'},
            {'text': 'Paint Shop', 'x': 32, 'y': 50, 'color': '#4ecdc4'},
            {'text': 'Engine Assembly', 'x': 72, 'y': 75, 'color': '#45b7d1'},
            {'text': 'Quality Control', 'x': 72, 'y': 50, 'color': '#96ceb4'},
            {'text': 'Control Center', 'x': 50, 'y': 25, 'color': '#feca57'}
        ]
        
        for zone in zone_labels:
            ax.text(zone['x'], zone['y'], zone['text'], ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=zone['color'],
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error in create_toyota_site_map: {e}")
        return None

def visualize_process_mapping(graph: nx.Graph, style: str = 'plant_floor') -> Optional[str]:
    """
    Visualize process mapping with different styles.
    
    Args:
        graph: NetworkX graph
        style: Visualization style ('plant_floor', '3d_plant_floor', 'network')
        
    Returns:
        Base64 encoded image string or None
    """
    try:
        if style == '3d_plant_floor':
            # 3D visualization
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get node positions
            pos = nx.spring_layout(graph, k=3, iterations=50)
            
            # Extract coordinates
            xs = [pos[node][0] for node in graph.nodes()]
            ys = [pos[node][1] for node in graph.nodes()]
            zs = [0.1] * len(graph.nodes())  # Slight elevation
            
            # Plot nodes
            ax.scatter(xs, ys, zs, c='lightblue', s=100, edgecolors='black')
            
            # Plot edges
            for edge in graph.edges():
                x = [pos[edge[0]][0], pos[edge[1]][0]]
                y = [pos[edge[0]][1], pos[edge[1]][1]]
                z = [0.1, 0.1]
                ax.plot(x, y, z, 'gray', alpha=0.6)
            
            # Add labels
            for node in graph.nodes():
                ax.text(pos[node][0], pos[node][1], 0.1, node, 
                       fontsize=8, ha='center', va='bottom')
            
            ax.set_title('3D Process Mapping')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        else:
            # 2D visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(graph, k=3, iterations=50)
            
            # Draw the graph
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
                   node_size=1000, font_size=8, font_weight='bold',
                   edge_color='gray', width=2, alpha=0.7, ax=ax)
            
            ax.set_title('Process Mapping')
        
        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error in process mapping visualization: {e}")
        return None

def make_globe(df) -> Optional[Any]:
    """
    Create a globe visualization for multi-site assets.
    
    Args:
        df: DataFrame with device information
        
    Returns:
        PyDeck chart or None
    """
    try:
        import pydeck as pdk
        
        # Sample data for globe visualization
        globe_data = [
            {"lat": 40.7128, "lon": -74.0060, "name": "New York Plant", "devices": 15},
            {"lat": 34.0522, "lon": -118.2437, "name": "LA Facility", "devices": 12},
            {"lat": 51.5074, "lon": -0.1278, "name": "London Site", "devices": 8}
        ]
        
        chart = pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=pdk.ViewState(
                latitude=40.7128,
                longitude=-74.0060,
                zoom=3,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=globe_data,
                    get_position=["lon", "lat"],
                    get_radius="devices",
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    radius_scale=6,
                    radius_min_pixels=1,
                    radius_max_pixels=100,
                    line_width_min_pixels=1,
                )
            ],
        )
        
        return chart
        
    except ImportError:
        print("PyDeck not available for globe visualization")
        return None
    except Exception as e:
        print(f"Error in globe visualization: {e}")
        return None


def analyze_graph_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze graph metrics for OT network.
    
    Args:
        G: NetworkX graph to analyze
        
    Returns:
        Dictionary with graph metrics
    """
    metrics = {
        'total_nodes': len(G.nodes),
        'total_edges': len(G.edges),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G) if len(G.nodes) > 1 else 0,
        'node_types': {},
        'latency_stats': {},
        'standards_distribution': {}
    }
    
    # Node type distribution
    for node in G.nodes:
        node_type = G.nodes[node].get('type', 'unknown')
        metrics['node_types'][node_type] = metrics['node_types'].get(node_type, 0) + 1
    
    # Latency statistics
    latencies = [G.nodes[node].get('latency', 0) for node in G.nodes]
    if latencies:
        metrics['latency_stats'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
    
    # Standards distribution
    for node in G.nodes:
        tags = G.nodes[node].get('standards_tags', {})
        for standard, value in tags.items():
            if standard not in metrics['standards_distribution']:
                metrics['standards_distribution'][standard] = {}
            if value not in metrics['standards_distribution'][standard]:
                metrics['standards_distribution'][standard][value] = 0
            metrics['standards_distribution'][standard][value] += 1
    
    return metrics 