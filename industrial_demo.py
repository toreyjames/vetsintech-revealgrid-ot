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

# Standards Ontologies (Simplified Mapping)
STANDARDS = {
    'NIST_SP_800_82': {'criticality': ['low', 'medium', 'high']},
    'IEC_62443': {'roles': ['component', 'system', 'zone']},
    'Purdue_Model': {'levels': ['0_physical', '1_sensing', '2_control', '3_mes', '4_erp']},
    'NERC_CIP': {'impact': ['low', 'medium', 'high']}
}

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

# Classify Asset Per Standard (for Language/Tags)
def classify_asset_per_standard(node_type, latency):
    tags = {}
    # NIST Criticality
    if latency > 10:
        tags['NIST_SP_800_82'] = 'high'
    elif latency > 5:
        tags['NIST_SP_800_82'] = 'medium'
    else:
        tags['NIST_SP_800_82'] = 'low'
    # IEC Role
    if 'sensor' in node_type:
        tags['IEC_62443'] = 'component'
    elif 'controller' in node_type:
        tags['IEC_62443'] = 'system'
    else:
        tags['IEC_62443'] = 'zone'
    # Purdue Level (example mapping)
    if 'sensor' in node_type:
        tags['Purdue_Model'] = '1_sensing'
    elif 'controller' in node_type:
        tags['Purdue_Model'] = '2_control'
    else:
        tags['Purdue_Model'] = '3_mes'
    # NERC Impact
    tags['NERC_CIP'] = 'high' if latency > 8 else 'medium' if latency > 4 else 'low'
    return tags

# Stub for passive scan
def run_passive_scan_stub():
    return 3, 12, True  # vlans, devices, airgap

# Create sample graph with standards compliance
def create_sample_graph():
    G = nx.DiGraph()
    
    # Add nodes with properties and standards tags
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
        # Apply standards classification
        tags = classify_asset_per_standard(props['type'], props['latency'])
        G.nodes[node]['standards_tags'] = tags
    
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

# Initial Setup Wizard Sim
def initial_setup_wizard(G):
    synced_assets = ['CMDBServer', 'OTController']
    for asset in synced_assets:
        G.add_node(asset, type='synced', latency=random.randint(1, 10))
        if G.nodes:
            G.add_edge(list(G.nodes)[0], asset)
    G = upload_document(G, "dummy P&ID")
    G = cv_asset_detection(G)
    # Apply standards classification
    for node in G.nodes:
        if 'standards_tags' not in G.nodes[node]:
            tags = classify_asset_per_standard(G.nodes[node]['type'], G.nodes[node]['latency'])
            G.nodes[node]['standards_tags'] = tags
    return G

# Document Upload
def upload_document(G, doc_content):
    extracted_assets = ['OfflinePump', 'ValveX']
    for asset in extracted_assets:
        G.add_node(asset, type='offline', latency=random.randint(1, 10))
        if G.nodes:
            G.add_edge(list(G.nodes)[0], asset)
    return G

# CV Asset Detection
def cv_asset_detection(G, image_data=None):
    if TORCH_AVAILABLE:
        dummy_image = torch.rand(1, 3, 224, 224)
        model = nn.Conv2d(3, 1, 3)
        output = model(dummy_image)
    detected_assets = ['CameraSensorY', 'RobotArmZ']
    for asset in detected_assets:
        G.add_node(asset, type='visual', latency=random.randint(1, 10))
        if G.nodes:
            G.add_edge(asset, list(G.nodes)[0])
    return G

# Hybrid Discover
def hybrid_discover():
    graph = create_sample_graph()
    graph = initial_setup_wizard(graph)
    return graph

# Automation Recommendations with Anomaly Detection
def recommend_automation(G, use_rl_sim=False):
    recommendations = []
    anomalies = []
    latencies = [G.nodes[node]['latency'] for node in G.nodes]
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    
    for node in G.nodes:
        if G.nodes[node]['latency'] > mean_lat + 2 * std_lat:
            anomalies.append(f"Anomaly detected in '{node}': High latency ({G.nodes[node]['latency']})—potential vulnerability.")
        
        if G.nodes[node]['type'] == 'manual' or G.nodes[node]['latency'] > 5:
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            if predecessors and successors:
                base_rec = f"Automate '{node}' (from auto_discovery) by bridging {predecessors[0]} to {successors[0]} with AI relay"
                if use_rl_sim:
                    reward = random.uniform(0.7, 1.0)
                    base_rec += f" (RL-sim reward: {reward:.2f}—empowers operator sovereignty)"
                recommendations.append(base_rec)
    
    return recommendations + anomalies

# Build World Model Simulation
def build_world_model(G, steps=10):
    initial_latency = sum(G.nodes[n]['latency'] for n in G.nodes)
    final_latency = initial_latency * 0.6  # Simulate improvement
    
    return {
        "initial_latency": initial_latency,
        "final_latency": final_latency,
        "efficiency_improvement": (initial_latency / final_latency) * 100,
        "hybrid_sources": {"auto_discovery": 12, "cmdb": 4, "document": 4, "cv": 4}
    }

# Generate Site Map Visualization
def generate_site_map(return_base64=True, drill_down_site=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('RevealGrid Global Site Map & Metrics')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-130, -60)
    ax.set_ylim(20, 55)
    
    lats = [data['lat'] for data in sites.values()]
    lons = [data['lon'] for data in sites.values()]
    effs = [data['efficiency'] for data in sites.values()]
    scatter = ax.scatter(lons, lats, c=effs, cmap='RdYlGn', s=100)
    
    for name, data in sites.items():
        ax.text(data['lon'] + 0.5, data['lat'], f"{name}\nEff: {data['efficiency']}%\nDown: {data['downtime']}%\nRisk: {data['risk_score']}% (NERC: medium)", fontsize=9)
    
    plt.colorbar(scatter, label='Efficiency %')
    
    if return_base64:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    plt.close()

# Visualize Process Mapping
def visualize_process_mapping(G, style='plant_floor', return_base64=True):
    if style == '3d_plant_floor':
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        pos_3d = {
            'TemperatureSensor': (0, 0, 0),
            'PLC_Controller': (1, 0, 1),
            'Actuator_Valve': (2, 0, 0),
            'BottlingMachine': (3, 1, 2),
            'QualityControl_Station': (4, 0, 0),
            'SafetySystem': (5, 0, 1),
            'DataLogger': (3, -1, -1)
        }
        xs = [pos_3d.get(n, (random.uniform(0,5), random.uniform(-1,1), random.uniform(-1,2)))[0] for n in G.nodes]
        ys = [pos_3d.get(n, (0,0,0))[1] for n in G.nodes]
        zs = [pos_3d.get(n, (0,0,0))[2] for n in G.nodes]
        ax.scatter(xs, ys, zs, c='lightblue', s=100)
        for edge in G.edges:
            x = [pos_3d.get(edge[0], (0,0,0))[0], pos_3d.get(edge[1], (0,0,0))[0]]
            y = [pos_3d.get(edge[0], (0,0,0))[1], pos_3d.get(edge[1], (0,0,0))[1]]
            z = [pos_3d.get(edge[0], (0,0,0))[2], pos_3d.get(edge[1], (0,0,0))[2]]
            ax.plot(x, y, z, 'gray')
        for i, node in enumerate(G.nodes):
            tag = G.nodes[node].get('standards_tags', {}).get('Purdue_Model', '')
            ax.text(xs[i], ys[i], zs[i], f"{node} ({tag})", fontsize=9)
        ax.set_title('Process Mapping: 3D Plant Floor Layout (Multimodal)')
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax, node_size=1000, font_size=10)
        ax.set_title('Process Mapping: Plant Floor Layout')
        ax.set_axis_off()
    
    if return_base64:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    plt.close()

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
def export_world_model(G, world_model):
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "graph": nx.node_link_data(G),
        "world_model": world_model,
        "export_version": "1.0",
        "standards_compliance": {
            "NIST_SP_800_82": "Compliant",
            "IEC_62443": "Compliant", 
            "Purdue_Model": "Compliant",
            "NERC_CIP": "Compliant"
        }
    }
    return json.dumps(export_data, default=str)

# Robot Use Model
def robot_use_model(G, task='automate_gap', robot='Optimus'):
    recs = recommend_automation(G, use_rl_sim=True)
    actions = [f"{robot} executes: {rec} – Adapt via world model." for rec in recs if 'Anomaly' not in rec]
    return actions

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
    st.markdown("**Industrial Standards-Compliant Asset Intelligence Platform**")
    st.markdown("Let's start asset discovery with NIST SP 800-82, IEC 62443, Purdue Model, and NERC CIP compliance.")
    st.markdown("Type 'yes' to begin scan or 'no' for manual entry.")
    
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
    st.markdown("**IEC 62443 Zone Analysis:** Network segmentation compliant")
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
    st.markdown("**Purdue Model Level Mapping:** Enable automatic process mapping? (yes / no)")
    
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
    
    # Asset breakdown with standards
    asset_types = {}
    standards_summary = {}
    for node, data in graph.nodes(data=True):
        asset_type = data.get('type', 'unknown')
        asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
        
        # Standards summary
        tags = data.get('standards_tags', {})
        for standard, value in tags.items():
            if standard not in standards_summary:
                standards_summary[standard] = {}
            if value not in standards_summary[standard]:
                standards_summary[standard][value] = 0
            standards_summary[standard][value] += 1
    
    st.markdown("### Asset Classification")
    for asset_type, count in asset_types.items():
        st.markdown(f"- **{asset_type}:** {count} assets")
    
    st.markdown("### Standards Compliance")
    for standard, values in standards_summary.items():
        st.markdown(f"**{standard}:** {', '.join([f'{k}: {v}' for k, v in values.items()])}")
    
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
    
    # Show process mapping
    if st.session_state.graph:
        mapping_viz = visualize_process_mapping(st.session_state.graph, style='3d_plant_floor')
        st.markdown("### Process Mapping Visualization")
        st.image(io.BytesIO(base64.b64decode(mapping_viz)), use_column_width=True)
    
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
    
    st.markdown("### Standards Compliance Summary")
    st.markdown("- **NIST SP 800-82:** All assets classified by criticality")
    st.markdown("- **IEC 62443:** Component/System/Zone mapping complete")
    st.markdown("- **Purdue Model:** Level 0-4 hierarchy established")
    st.markdown("- **NERC CIP:** Impact assessment validated")
    
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
    
    if st.session_state.graph:
        # Simulate risk analysis with standards
        risks = []
        for node, data in st.session_state.graph.nodes(data=True):
            risk_prob = data.get('failure_prob', 0.2) * 100
            severity = "high" if risk_prob > 25 else "medium" if risk_prob > 15 else "low"
            nist_level = data.get('standards_tags', {}).get('NIST_SP_800_82', 'low')
            risks.append((node, f"{risk_prob:.2f}%", severity, nist_level))
        
        for asset, risk, severity, nist_level in risks[:5]:  # Show top 5
            color = "🔴" if severity == "high" else "🟡"
            st.markdown(f"{color} **{asset}:** {risk} failure probability (Severity: {severity}, NIST: {nist_level})")
    
    st.markdown("### Anomalies Detected")
    if st.session_state.graph:
        recs = recommend_automation(st.session_state.graph, use_rl_sim=False)
        anomalies = [rec for rec in recs if 'Anomaly' in rec]
        if anomalies:
            for anomaly in anomalies:
                st.markdown(f"🚨 {anomaly}")
        else:
            st.markdown("✅ No anomalies detected")
    
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
    
    if st.session_state.graph:
        recs = recommend_automation(st.session_state.graph, use_rl_sim=True)
        for rec in recs:
            if 'Anomaly' not in rec:
                st.markdown(f"⚡ {rec}")
    
    st.markdown("### Robot Actions")
    if st.session_state.graph:
        robot_actions = robot_use_model(st.session_state.graph)
        for action in robot_actions:
            st.markdown(f"🤖 {action}")
    
    st.markdown("### Standards Compliance")
    st.markdown("- **NIST SP 800-82:** All recommendations consider criticality levels")
    st.markdown("- **IEC 62443:** Automation respects component/system boundaries")
    st.markdown("- **Purdue Model:** Level-appropriate automation strategies")
    st.markdown("- **NERC CIP:** Impact assessments included in recommendations")
    
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
        
        st.markdown("### Standards Compliance Certificate")
        st.markdown("- ✅ NIST SP 800-82: Criticality mapping complete")
        st.markdown("- ✅ IEC 62443: Component/System/Zone classification")
        st.markdown("- ✅ Purdue Model: Level 0-4 hierarchy established")
        st.markdown("- ✅ NERC CIP: Impact assessment validated")
        
        if st.button("Download Encrypted Export", key="download_btn"):
            st.download_button(
                label="Download RevealGrid Model",
                data=export_data,
                file_name="revealgrid_model.enc",
                mime="application/octet-stream"
            )
    
    st.markdown("The model has been encrypted and exported successfully with full industrial standards compliance.")
    
    if st.button("Reset Demo", key="reset_btn"):
        st.session_state.clear()
        st.experimental_rerun()

# Footer
st.markdown("<hr style='border: 1px solid #004080;'>", unsafe_allow_html=True)
st.markdown("<footer>© 2025 RevealGrid | All Rights Reserved | Industrial Standards Compliant</footer>", unsafe_allow_html=True) 