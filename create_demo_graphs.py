#!/usr/bin/env python3
"""
Create demo graph files for RevealGrid OT demo sites.
"""

import networkx as nx
import pickle
from pathlib import Path

def create_detroit_graph():
    """Create Detroit Engine Plant process graph."""
    G = nx.DiGraph()
    
    # Engine assembly process nodes
    nodes = [
        'PLC_Engine_Control', 'HMI_Engine_Monitor', 'Robot_Engine_Assembly',
        'Sensor_Cylinder_Head', 'Actuator_Valve_Control', 'QC_Engine_Test',
        'PLC_Fuel_System', 'HMI_Fuel_Monitor', 'Sensor_Fuel_Pressure',
        'Actuator_Fuel_Pump', 'Robot_Fuel_Injector', 'QC_Fuel_System'
    ]
    
    G.add_nodes_from(nodes)
    
    # Engine assembly process flow
    edges = [
        ('PLC_Engine_Control', 'HMI_Engine_Monitor'),
        ('HMI_Engine_Monitor', 'Robot_Engine_Assembly'),
        ('Robot_Engine_Assembly', 'Sensor_Cylinder_Head'),
        ('Sensor_Cylinder_Head', 'Actuator_Valve_Control'),
        ('Actuator_Valve_Control', 'QC_Engine_Test'),
        ('QC_Engine_Test', 'PLC_Fuel_System'),
        ('PLC_Fuel_System', 'HMI_Fuel_Monitor'),
        ('HMI_Fuel_Monitor', 'Sensor_Fuel_Pressure'),
        ('Sensor_Fuel_Pressure', 'Actuator_Fuel_Pump'),
        ('Actuator_Fuel_Pump', 'Robot_Fuel_Injector'),
        ('Robot_Fuel_Injector', 'QC_Fuel_System')
    ]
    
    G.add_edges_from(edges)
    return G

def create_austin_graph():
    """Create Austin Fabrication Plant process graph."""
    G = nx.DiGraph()
    
    # Fabrication process nodes
    nodes = [
        'PLC_Fab_Control', 'HMI_Fab_Monitor', 'Robot_Welding',
        'Sensor_Material_Thickness', 'Actuator_Press_Control', 'QC_Weld_Quality',
        'PLC_Cutting_Control', 'HMI_Cutting_Monitor', 'Robot_Cutting',
        'Sensor_Cut_Dimensions', 'Actuator_Conveyor', 'QC_Cutting_Quality'
    ]
    
    G.add_nodes_from(nodes)
    
    # Fabrication process flow
    edges = [
        ('PLC_Fab_Control', 'HMI_Fab_Monitor'),
        ('HMI_Fab_Monitor', 'Robot_Welding'),
        ('Robot_Welding', 'Sensor_Material_Thickness'),
        ('Sensor_Material_Thickness', 'Actuator_Press_Control'),
        ('Actuator_Press_Control', 'QC_Weld_Quality'),
        ('QC_Weld_Quality', 'PLC_Cutting_Control'),
        ('PLC_Cutting_Control', 'HMI_Cutting_Monitor'),
        ('HMI_Cutting_Monitor', 'Robot_Cutting'),
        ('Robot_Cutting', 'Sensor_Cut_Dimensions'),
        ('Sensor_Cut_Dimensions', 'Actuator_Conveyor'),
        ('Actuator_Conveyor', 'QC_Cutting_Quality')
    ]
    
    G.add_edges_from(edges)
    return G

def create_kentucky_graph():
    """Create Kentucky Plant process graph."""
    G = nx.DiGraph()
    
    # Kentucky plant process nodes
    nodes = [
        'PLC_KY001', 'HMI_KY02', 'VFD_KY07', 'ROBOT_KY01', 'SENSOR_KY03',
        'PLC_KY002', 'HMI_KY03', 'ROBOT_KY02', 'SENSOR_KY04', 'PLC_KY003'
    ]
    
    G.add_nodes_from(nodes)
    
    # Kentucky plant process flow
    edges = [
        ('PLC_KY001', 'HMI_KY02'),
        ('HMI_KY02', 'SENSOR_KY03'),
        ('SENSOR_KY03', 'PLC_KY002'),
        ('PLC_KY002', 'HMI_KY03'),
        ('HMI_KY03', 'ROBOT_KY02'),
        ('ROBOT_KY02', 'SENSOR_KY04'),
        ('SENSOR_KY04', 'PLC_KY003'),
        ('PLC_KY003', 'VFD_KY07'),
        ('VFD_KY07', 'ROBOT_KY01'),
        ('ROBOT_KY01', 'PLC_KY001')
    ]
    
    G.add_edges_from(edges)
    return G

def main():
    """Create all demo graph files."""
    demo_dir = Path("data/demo_sites")
    demo_dir.mkdir(exist_ok=True)
    
    # Create Detroit graph
    detroit_graph = create_detroit_graph()
    with open(demo_dir / "detroit_graph.gpickle", "wb") as f:
        pickle.dump(detroit_graph, f)
    print("Created detroit_graph.gpickle")
    
    # Create Austin graph
    austin_graph = create_austin_graph()
    with open(demo_dir / "austin_graph.gpickle", "wb") as f:
        pickle.dump(austin_graph, f)
    print("Created austin_graph.gpickle")
    
    # Create Kentucky graph
    kentucky_graph = create_kentucky_graph()
    with open(demo_dir / "ky_graph.gpickle", "wb") as f:
        pickle.dump(kentucky_graph, f)
    print("Created ky_graph.gpickle")
    
    print("All demo graph files created successfully!")

if __name__ == "__main__":
    main() 