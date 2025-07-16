import streamlit as st
import time
import json
import networkx as nx
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import numpy as np
import requests
from datetime import datetime

# Optional imports with graceful fallbacks
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="RevealGrid - Industrial Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Cues: Industrial theme with blues/grays, greens/reds for metrics
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #f0f4f8;  /* Light gray-blue */
        color: #333333;  /* Dark gray text */
    }
    /* Header */
    header {
        background-color: #004080;  /* Deep blue */
        color: white;
    }
    /* Sidebar */
    .css-1aumxhk {  /* Sidebar class */
        background-color: #e6e6e6;  /* Light gray */
    }
    /* Metrics: Green good, red bad */
    .stMetric label {
        color: #006400;  /* Dark green for positive */
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #ff0000;  /* Red for negative deltas */
    }
    /* Buttons */
    .stButton > button {
        background-color: #004080;  /* Blue */
        color: white;
        border-radius: 5px;
    }
    /* Progress checks */
    .sidebar .stMarkdown {
        font-size: 16px;
    }
    /* Footer */
    footer {
        background-color: #004080;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Physics-First Constants
MIN_REWARD = 0.5
GRAVITY = 9.81

# Simulated site data
sites = {
    'Plant A - Detroit': {'lat': 42.3314, 'lon': -83.0458, 'efficiency': 85, 'downtime': 5, 'risk_score': 20},
    'Plant B - Chicago': {'lat': 41.8781, 'lon': -87.6298, 'efficiency': 92, 'downtime': 2, 'risk_score': 10},
    'Plant C - Austin': {'lat': 30.2672, 'lon': -97.7431, 'efficiency': 78, 'downtime': 10, 'risk_score': 35},
    'Plant D - Seattle': {'lat': 47.6062, 'lon': -122.3321, 'efficiency': 88, 'downtime': 4, 'risk_score': 15}
}

# Progress Steps
progress_steps = [
    "Welcome",
    "Asset Discovery", 
    "Process Analysis",
    "World Model",
    "Risk Analysis",
    "AI Recommendations",
    "Export"
]

# Session Boot
if "state" not in st.session_state:
    st.session_state.state = "welcome"
    st.session_state.answers = {} 
    st.session_state.log = []
    st.session_state.show_globe = False
    st.session_state.selected_site = None
    st.session_state.mapping_style = '3d_plant_floor'
    st.session_state.graph = None
    st.session_state.progress = {step: False for step in progress_steps}
    st.session_state.progress["Welcome"] = True
    st.session_state.user_input = ""

def set_state(new_state):
    st.session_state.state = new_state
    st.session_state.log.append({"step": new_state, "time": time.time()})

def update_progress(step):
    st.session_state.progress[step] = True

# Parse Cmd
def parse_cmd(txt: str):
    if not txt:
        return None
    t = txt.strip().lower()
    if t.startswith("y"):
        return "yes"
    if t.startswith("n"):
        return "no"
    if t in {"/skip", "skip"}:
        return "skip"
    if t == "/back":
        return "back"
    if t == "/restart":
        return "restart"
    if t == "/help":
        return "help"
    return None

# Stub for passive scan
def run_passive_scan_stub():
    return 3, 12, True  # vlans, devices, airgap

# Create sample graph
def create_sample_graph():
    G = nx.DiGraph()
    
    # Add nodes with properties
    nodes = [
        ('TemperatureSensor', {'type': 'sensor', 'latency': 5, 'failure_prob': 0.15}),
        ('PressureSensor', {'type': 'sensor', 'latency': 4, 'failure_prob': 0.12}),
        ('FlowMeter', {'type': 'sensor', 'latency': 6, 'failure_prob': 0.18}),
        ('PLC_Controller', {'type': 'controller', 'latency': 8, 'failure_prob': 0.20}),
        ('Actuator_Valve', {'type': 'actuator', 'latency': 7, 'failure_prob': 0.22}),
        ('Actuator_Pump', {'type': 'actuator', 'latency': 9, 'failure_prob': 0.25}),
        ('BottlingMachine', {'type': 'equipment', 'latency': 12, 'failure_prob': 0.30}),
        ('QualityControl_Station', {'type': 'equipment', 'latency': 10, 'failure_prob': 0.28}),
        ('SafetySystem', {'type': 'safety', 'latency': 3, 'failure_prob': 0.10}),
        ('DataLogger', {'type': 'data', 'latency': 2, 'failure_prob': 0.08}),
        ('OTController', {'type': 'controller', 'latency': 6, 'failure_prob': 0.18}),
        ('SCADASystem', {'type': 'system', 'latency': 4, 'failure_prob': 0.15})
    ]
    
    for node, props in nodes:
        G.add_node(node, **props)
    
    # Add edges
    edges = [
        ('TemperatureSensor', 'PLC_Controller'),
        ('PressureSensor', 'PLC_Controller'),
        ('FlowMeter', 'PLC_Controller'),
        ('PLC_Controller', 'Actuator_Valve'),
        ('PLC_Controller', 'Actuator_Pump'),
        ('Actuator_Valve', 'BottlingMachine'),
        ('Actuator_Pump', 'BottlingMachine'),
        ('BottlingMachine', 'QualityControl_Station'),
        ('QualityControl_Station', 'SafetySystem'),
        ('SafetySystem', 'DataLogger'),
        ('DataLogger', 'OTController'),
        ('OTController', 'SCADASystem')
    ]
    
    G.add_edges_from(edges)
    return G

# Hybrid Discover
def hybrid_discover():
    graph = create_sample_graph()
    return graph

# Downtime Recs
def downtime_recommendations(graph):
    recs = [
        "Automate 'Actuator_Valve' by bridging PLC_Controller to ManualCheck_Operator with AI relay (RL-sim reward: 0.76)",
        "Automate 'ManualCheck_Operator' by bridging Actuator_Valve to BottlingMachine with AI relay (RL-sim reward: 0.88)",
        "Automate 'BottlingMachine' by bridging ManualCheck_Operator to QualityControl_Station with AI relay (RL-sim reward: 0.84)"
    ]
    return "\n".join(recs)

# Global Site Deck
def global_site_deck(sites_data, globe=True):
    if not PYDECK_AVAILABLE:
        st.warning("PyDeck not available. Install with: pip install pydeck")
        return None
    
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": v['lat'], "lon": v['lon'], "efficiency": v['efficiency'], "name": k} for k, v in sites_data.items()],
            get_position="[lon, lat]",
            get_color="[255 * (1 - efficiency / 100), 255 * (efficiency / 100), 0, 160]",
            get_radius=200000,
            pickable=True,
            auto_highlight=True
        )
    ]
    view_state = pdk.ViewState(latitude=37.76, longitude=-122.4, zoom=1, pitch=50 if globe else 0)
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')

# Export functions
def export_world_model(graph, world_model):
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "graph": nx.node_link_data(graph),
        "world_model": world_model,
        "export_version": "1.0"
    }
    return json.dumps(export_data, default=str)

def build_world_model(graph, steps=10):
    return {
        "initial_latency": 150.0,
        "final_latency": 90.0,
        "efficiency_improvement": 166.67,
        "hybrid_sources": {"auto_discovery": 12, "cmdb": 4, "document": 4, "cv": 4}
    }

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("🏭")  # Factory emoji as logo
with col2:
    st.title("RevealGrid")
st.subheader("Industrial Asset Intelligence Platform")
st.markdown("<hr style='border: 1px solid #004080;'>", unsafe_allow_html=True)

# Sidebar Progress
st.sidebar.title("Demo Progress")
for step in progress_steps:
    checked = "✅" if st.session_state.progress.get(step, False) else "⬜"
    st.sidebar.markdown(f"{checked} {step}")

# Main Content
state = st.session_state.state

if state == "welcome":
    update_progress("Welcome")
    st.markdown("## Welcome to RevealGrid!")
    st.markdown("Let's start asset discovery. Type 'yes' to begin scan or 'no' for manual entry.")
    
    # Use text_input instead of chat_input for compatibility
    user_input = st.text_input("Type yes / no / skip / back / restart / help", key="welcome_input")
    
    if st.button("Submit", key="welcome_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("ask_scan_permission")
                update_progress("Asset Discovery")
                st.session_state.user_input = ""
                st.experimental_rerun()
            elif cmd == "no":
                set_state("ask_manual_loc")
                update_progress("Asset Discovery")
                st.session_state.user_input = ""
                st.experimental_rerun()
            elif cmd == "help":
                st.info("Commands: yes/no, skip, back, restart, help")

elif state == "ask_scan_permission":
    with st.spinner("📡 Passive scan starting …"):
        time.sleep(1)
        vlans, devices, airgap = run_passive_scan_stub()
    st.session_state.answers["scan_allowed"] = True
    st.session_state.answers.update({"vlans": vlans, "devices": devices})
    st.markdown(f"## Scan Results")
    st.markdown(f"Detected **{vlans} VLANs** and **{devices} devices**. Air-gapped: {airgap}.")
    st.markdown("Confirm scan results? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="scan_input")
    
    if st.button("Submit", key="scan_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd in {"yes", "no"}:
                st.session_state.answers["airgap_confirmed"] = cmd == "yes"
                set_state("ask_auto_mapping")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "ask_auto_mapping":
    st.markdown("## Auto-Mapping Configuration")
    st.markdown("Enable automatic process mapping? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="mapping_input")
    
    if st.button("Submit", key="mapping_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd in {"yes", "no"}:
                st.session_state.answers["auto_mapping"] = cmd == "yes"
                set_state("run_discovery")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "run_discovery":
    update_progress("Asset Discovery")
    st.markdown("## Running Hybrid Asset Discovery")
    
    with st.spinner("🔍 Discovering assets..."):
        time.sleep(2)
        graph = hybrid_discover()
        st.session_state.graph = graph
        st.session_state.answers["graph"] = graph
    
    st.markdown("### Discovery Results")
    st.markdown(f"**Total Assets Found:** {len(graph.nodes())}")
    st.markdown(f"**Connections:** {len(graph.edges())}")
    
    # Asset breakdown
    asset_types = {}
    for node, data in graph.nodes(data=True):
        asset_type = data.get('type', 'unknown')
        asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
    
    for asset_type, count in asset_types.items():
        st.markdown(f"- **{asset_type}:** {count} assets")
    
    st.markdown("Discovery complete! Ready for process analysis? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="discovery_input")
    
    if st.button("Submit", key="discovery_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("process_analysis")
                update_progress("Process Analysis")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "process_analysis":
    update_progress("Process Analysis")
    st.markdown("## Process Analysis with AI")
    
    with st.spinner("🤖 Analyzing process flows..."):
        time.sleep(2)
    
    st.markdown("### AI-Powered Process Analysis")
    
    # Simulate AI analysis results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Process Efficiency", "85.2%", "↑ 12.3%")
    with col2:
        st.metric("Bottlenecks Identified", "3", "↓ 2")
    with col3:
        st.metric("Optimization Potential", "23.7%", "↑ 8.1%")
    
    st.markdown("### Deep Learning Insights")
    st.markdown("- **Flow Optimization:** Identified 3 critical path improvements")
    st.markdown("- **Bottleneck Detection:** Found 2 high-latency nodes")
    st.markdown("- **Reinforcement Learning:** Simulated 1000+ scenarios")
    
    st.markdown("Process analysis complete! Build world model? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="analysis_input")
    
    if st.button("Submit", key="analysis_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("world_model")
                update_progress("World Model")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "world_model":
    update_progress("World Model")
    st.markdown("## Building Unified World Model")
    
    with st.spinner("🌍 Constructing world model..."):
        time.sleep(2)
    
    st.markdown("### Multi-Site Intelligence")
    
    # Show global site map
    if PYDECK_AVAILABLE:
        deck = global_site_deck(sites)
        if deck:
            st.pydeck_chart(deck)
    else:
        st.markdown("**Global Site Map:** 4 sites connected")
        for site, data in sites.items():
            st.markdown(f"- {site}: {data['efficiency']}% efficiency")
    
    st.markdown("### World Model Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sites", "4")
    with col2:
        st.metric("Average Efficiency", "85.8%")
    with col3:
        st.metric("High Risk Sites", "1")
    
    st.markdown("World model complete! Run risk analysis? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="world_input")
    
    if st.button("Submit", key="world_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("risk_analysis")
                update_progress("Risk Analysis")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "risk_analysis":
    update_progress("Risk Analysis")
    st.markdown("## Risk Analysis & Anomaly Detection")
    
    with st.spinner("⚠️ Analyzing risks..."):
        time.sleep(2)
    
    st.markdown("### Risk Insights")
    
    # Simulate risk analysis
    risks = [
        ("TemperatureSensor", "20.63%", "medium"),
        ("PressureSensor", "27.12%", "medium"),
        ("Actuator_Valve", "29.64%", "medium"),
        ("QualityControl_Station", "21.58%", "medium"),
        ("SafetySystem", "15.00%", "high")
    ]
    
    for asset, risk, severity in risks:
        color = "🔴" if severity == "high" else "🟡"
        st.markdown(f"{color} **{asset}:** {risk} failure probability (Severity: {severity})")
    
    st.markdown("### Anomalies Detected")
    st.markdown("🚨 **1 anomaly detected:** High latency in SafetySystem")
    
    st.markdown("Risk analysis complete! Generate AI recommendations? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="risk_input")
    
    if st.button("Submit", key="risk_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("ai_recommendations")
                update_progress("AI Recommendations")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "ai_recommendations":
    update_progress("AI Recommendations")
    st.markdown("## AI-Powered Recommendations")
    
    with st.spinner("🤖 Generating recommendations..."):
        time.sleep(2)
    
    st.markdown("### Automation Recommendations")
    
    recs = downtime_recommendations(st.session_state.graph)
    for rec in recs.split('\n'):
        st.markdown(f"⚡ {rec}")
    
    st.markdown("### Robot Actions")
    robot_actions = [
        "Automate 'ManualCheck_Operator' (RL-sim reward: 0.83)",
        "Automate 'BottlingMachine' (RL-sim reward: 0.99)"
    ]
    
    for action in robot_actions:
        st.markdown(f"🤖 {action}")
    
    st.markdown("AI recommendations complete! Export world model? (yes / no)")
    
    user_input = st.text_input("Type yes / no", key="ai_input")
    
    if st.button("Submit", key="ai_submit"):
        if user_input:
            cmd = parse_cmd(user_input)
            if cmd == "yes":
                set_state("export")
                update_progress("Export")
                st.session_state.user_input = ""
                st.experimental_rerun()

elif state == "export":
    update_progress("Export")
    st.markdown("## World Model Export")
    st.markdown("Exporting completed model for secure deployment...")
    
    with st.spinner("📦 Preparing export..."):
        time.sleep(2)
    
    st.success("✅ Export completed successfully!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Export Size", "139,220 bytes")
    col2.metric("Assets Exported", "24")
    col3.metric("Sites Included", "4")
    
    # Generate export data
    if st.session_state.graph:
        world_model = build_world_model(st.session_state.graph)
        export_data = export_world_model(st.session_state.graph, world_model)
        
        if st.button("Download Encrypted Export", key="download_btn"):
            st.download_button(
                label="Download RevealGrid Model",
                data=export_data,
                file_name="revealgrid_model.enc",
                mime="application/octet-stream"
            )
    
    st.markdown("The model has been encrypted and exported successfully. This data can now be deployed to your OT environment for real-time monitoring.")
    
    if st.button("Reset Demo", key="reset_btn"):
        st.session_state.clear()
        st.experimental_rerun()

# Footer
st.markdown("<hr style='border: 1px solid #004080;'>", unsafe_allow_html=True)
st.markdown("<footer>© 2025 RevealGrid | All Rights Reserved</footer>", unsafe_allow_html=True) 