"""
RevealGrid OT - Main Entry Point
Alternative entry point for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

# Enhanced site data with detailed assets
sites = [
    {
        'name': 'Plant A - Detroit', 
        'lat': 42.3314, 
        'lon': -83.0458, 
        'efficiency': 85,
        'assets': {
            'PLC_Controllers': 12,
            'HMI_Systems': 8,
            'Robots': 24,
            'Sensors': 156,
            'Actuators': 89,
            'Safety_Systems': 15,
            'QC_Stations': 6,
            'Conveyors': 18
        },
        'processes': ['Body_Assembly', 'Paint_Shop', 'Final_Assembly', 'Quality_Control'],
        'automation_level': 92,
        'downtime_hours': 5.2
    },
    {
        'name': 'Plant B - Chicago', 
        'lat': 41.8781, 
        'lon': -87.6298, 
        'efficiency': 92,
        'assets': {
            'PLC_Controllers': 15,
            'HMI_Systems': 12,
            'Robots': 32,
            'Sensors': 203,
            'Actuators': 112,
            'Safety_Systems': 22,
            'QC_Stations': 8,
            'Conveyors': 24
        },
        'processes': ['Stamping', 'Body_Assembly', 'Paint_Shop', 'Final_Assembly', 'Quality_Control'],
        'automation_level': 95,
        'downtime_hours': 2.1
    },
    {
        'name': 'Plant C - Austin', 
        'lat': 30.2672, 
        'lon': -97.7431, 
        'efficiency': 78,
        'assets': {
            'PLC_Controllers': 8,
            'HMI_Systems': 6,
            'Robots': 16,
            'Sensors': 98,
            'Actuators': 67,
            'Safety_Systems': 12,
            'QC_Stations': 4,
            'Conveyors': 14
        },
        'processes': ['Body_Assembly', 'Paint_Shop', 'Final_Assembly'],
        'automation_level': 78,
        'downtime_hours': 10.5
    },
    {
        'name': 'Kentucky Plant', 
        'lat': 37.8393, 
        'lon': -84.2700, 
        'efficiency': 88,
        'assets': {
            'PLC_Controllers': 18,
            'HMI_Systems': 14,
            'Robots': 28,
            'Sensors': 187,
            'Actuators': 95,
            'Safety_Systems': 18,
            'QC_Stations': 7,
            'Conveyors': 20
        },
        'processes': ['Stamping', 'Body_Assembly', 'Paint_Shop', 'Final_Assembly', 'Quality_Control', 'Testing'],
        'automation_level': 89,
        'downtime_hours': 3.8
    }
]

def create_process_graph(plant_name):
    """Create a process flow graph for the selected plant."""
    G = nx.DiGraph()
    
    # Get plant data
    plant = next((s for s in sites if s['name'] == plant_name), None)
    if not plant:
        return G
    
    # Add process nodes
    processes = plant['processes']
    for i, process in enumerate(processes):
        G.add_node(process, pos=(i, 0))
    
    # Add edges between processes
    for i in range(len(processes) - 1):
        G.add_edge(processes[i], processes[i + 1])
    
    return G

def visualize_process_flow(graph, plant_name):
    """Create a process flow visualization."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use spring layout for better positioning
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw the graph
        nx.draw(graph, pos, with_labels=True, 
                node_color='lightblue', 
                node_size=3000, 
                font_size=10, 
                font_weight='bold',
                arrows=True, 
                edge_color='gray', 
                arrowsize=20,
                connectionstyle="arc3,rad=0.1")
        
        plt.title(f"Process Flow - {plant_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def show_plant_details(plant_name):
    """Display detailed information for the selected plant."""
    plant = next((s for s in sites if s['name'] == plant_name), None)
    if not plant:
        st.error("Plant not found!")
        return
    
    st.markdown("---")
    st.header(f"{plant_name} - Detailed Analysis")
    
    # Back button
    if st.button("Back to Global Map"):
        st.session_state.selected_plant = None
        st.rerun()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Efficiency", f"{plant['efficiency']}%")
    with col2:
        st.metric("Automation Level", f"{plant['automation_level']}%")
    with col3:
        st.metric("Downtime", f"{plant['downtime_hours']}h")
    with col4:
        total_assets = sum(plant['assets'].values())
        st.metric("Total Assets", total_assets)
    
    # Asset breakdown
    st.subheader("Asset Inventory")
    
    # Create asset DataFrame for better display
    asset_df = pd.DataFrame([
        {'Asset Type': k.replace('_', ' '), 'Count': v}
        for k, v in plant['assets'].items()
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(asset_df, use_container_width=True)
    
    with col2:
        # Asset pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(asset_df['Count'], labels=asset_df['Asset Type'], autopct='%1.1f%%')
        ax.set_title('Asset Distribution')
        st.pyplot(fig)
    
    # Process flow
    st.subheader("Process Flow")
    graph = create_process_graph(plant_name)
    if graph.nodes():
        viz_base64 = visualize_process_flow(graph, plant_name)
        if viz_base64:
            st.image(base64.b64decode(viz_base64), use_container_width=True)
    
    # Automation recommendations
    st.subheader("Automation Recommendations")
    
    recommendations = []
    if plant['automation_level'] < 90:
        recommendations.append("Increase automation in manual processes")
    if plant['downtime_hours'] > 5:
        recommendations.append("Implement predictive maintenance")
    if plant['assets']['Sensors'] < 150:
        recommendations.append("Add more sensors for better monitoring")
    if plant['assets']['Safety_Systems'] < 15:
        recommendations.append("Enhance safety systems")
    
    if not recommendations:
        recommendations.append("Plant is well-optimized!")
    
    for rec in recommendations:
        st.write(f"- {rec}")
    
    # Asset health status
    st.subheader("Asset Health Status")
    
    # Simulate asset health data
    health_data = {
        'Excellent': 0.6,
        'Good': 0.25,
        'Fair': 0.1,
        'Poor': 0.05
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Excellent", f"{health_data['Excellent']*100:.0f}%", delta="+2%")
    with col2:
        st.metric("Good", f"{health_data['Good']*100:.0f}%", delta="-1%")
    with col3:
        st.metric("Fair", f"{health_data['Fair']*100:.0f}%", delta="0%")
    with col4:
        st.metric("Poor", f"{health_data['Poor']*100:.0f}%", delta="-1%")

def main():
    st.set_page_config(
        page_title="RevealGrid MVP Demo",
        page_icon="🏭",
        layout="wide"
    )
    
    # Initialize session state
    if 'selected_plant' not in st.session_state:
        st.session_state.selected_plant = None
    
    st.title("RevealGrid MVP Demo")
    st.markdown("**Interactive Asset Discovery & Process Understanding Platform**")

    # Debug info
    st.write("Debug: App is running")
    st.write(f"Number of sites: {len(sites)}")
    
    # Create DataFrame for map
    df = pd.DataFrame(sites)
    st.write("Site data:", df[['name', 'lat', 'lon', 'efficiency']])
    
    # Try Streamlit's built-in map (most reliable)
    st.subheader("Interactive Plant Map")
    try:
        st.map(df)
        st.success("Interactive map loaded!")
    except Exception as e:
        st.error(f"Streamlit map error: {e}")
    
    # Plant selection
    st.subheader("Select a Plant for Detailed Analysis")
    selected = st.radio(
        "Choose a plant to view detailed assets and processes:",
        options=[site['name'] for site in sites],
        key="plant_selector"
    )
    
    if selected:
        st.session_state.selected_plant = selected
        show_plant_details(selected)
    else:
        # Show site list as backup
        st.subheader("Available Plants")
        for site in sites:
            st.write(f"- {site['name']}: {site['efficiency']}% efficiency")
    
    # Global metrics
    st.subheader("Global Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sites", len(sites))
    with col2:
        avg_eff = np.mean([s['efficiency'] for s in sites])
        st.metric("Avg Efficiency", f"{avg_eff:.1f}%")
    with col3:
        st.metric("High Efficiency Sites", sum(1 for s in sites if s['efficiency'] >= 85))

if __name__ == "__main__":
    main() 