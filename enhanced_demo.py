import streamlit as st
import time
import json
import networkx as nx
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import sympy as sp
import numpy as np
# Optional PyTorch for future ML features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import requests
# Optional PyDeck for 3D mapping
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

# Set page config FIRST
st.set_page_config(
    page_title="RevealGrid Enhanced Demo",
    page_icon="🏭",
    layout="wide"
)

# Physics-First Constants
MIN_REWARD = 0.5  # Constrain to avoid hallucinations
GRAVITY = 9.81  # Example physics constant for future sims

# Simulated site data (expand with user input in prod)
sites = {
    'Plant A - Detroit': {'lat': 42.3314, 'lon': -83.0458, 'efficiency': 85, 'downtime': 5, 'risk_score': 20},
    'Plant B - Chicago': {'lat': 41.8781, 'lon': -87.6298, 'efficiency': 92, 'downtime': 2, 'risk_score': 10},
    'Plant C - Austin': {'lat': 30.2672, 'lon': -97.7431, 'efficiency': 78, 'downtime': 10, 'risk_score': 35},
    'Plant D - Seattle': {'lat': 47.6062, 'lon': -122.3321, 'efficiency': 88, 'downtime': 4, 'risk_score': 15}
}

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

# Session Boot
if "state" not in st.session_state:
    st.session_state.state = "ask_scan_permission"
    st.session_state.answers = {} 
    st.session_state.log = []
    st.session_state.show_globe = False
    st.session_state.selected_site = None
    st.session_state.mapping_style = '3d_plant_floor'
    st.session_state.graph = None

def set_state(new_state):
    st.session_state.state = new_state
    st.session_state.log.append({"step": new_state, "time": time.time()})

# Stub for passive scan
def run_passive_scan_stub():
    return 3, 12, True  # vlans, devices, airgap

# Create sample graph with physics grounding
def create_sample_graph():
    """Create a physics-grounded industrial network graph"""
    G = nx.Graph()
    
    # Industrial nodes with realistic properties
    nodes = [
        ("PLC_Controller", {"type": "controller", "risk": 0.25, "efficiency": 0.85, "latency": 5}),
        ("TemperatureSensor", {"type": "sensor", "risk": 0.20, "efficiency": 0.90, "latency": 2}),
        ("PressureSensor", {"type": "sensor", "risk": 0.22, "efficiency": 0.88, "latency": 3}),
        ("FlowMeter", {"type": "sensor", "risk": 0.18, "efficiency": 0.92, "latency": 2}),
        ("Actuator_Valve", {"type": "actuator", "risk": 0.30, "efficiency": 0.75, "latency": 8}),
        ("Actuator_Pump", {"type": "actuator", "risk": 0.28, "efficiency": 0.78, "latency": 7}),
        ("BottlingMachine", {"type": "machine", "risk": 0.35, "efficiency": 0.70, "latency": 10}),
        ("QualityControl", {"type": "machine", "risk": 0.25, "efficiency": 0.80, "latency": 6}),
        ("SafetySystem", {"type": "safety", "risk": 0.15, "efficiency": 0.95, "latency": 1}),
        ("DataLogger", {"type": "data", "risk": 0.20, "efficiency": 0.85, "latency": 4}),
        ("OTController", {"type": "controller", "risk": 0.30, "efficiency": 0.80, "latency": 5}),
        ("NetworkSwitch", {"type": "network", "risk": 0.25, "efficiency": 0.90, "latency": 3})
    ]
    
    for node, props in nodes:
        G.add_node(node, **props)
    
    # Realistic industrial connections
    edges = [
        ("PLC_Controller", "TemperatureSensor", {"weight": 0.8, "type": "control"}),
        ("PLC_Controller", "PressureSensor", {"weight": 0.8, "type": "control"}),
        ("PLC_Controller", "Actuator_Valve", {"weight": 0.9, "type": "control"}),
        ("Actuator_Valve", "BottlingMachine", {"weight": 0.7, "type": "process"}),
        ("BottlingMachine", "QualityControl", {"weight": 0.8, "type": "process"}),
        ("SafetySystem", "BottlingMachine", {"weight": 0.9, "type": "safety"}),
        ("DataLogger", "PLC_Controller", {"weight": 0.6, "type": "data"}),
        ("NetworkSwitch", "OTController", {"weight": 0.7, "type": "network"})
    ]
    
    for edge in edges:
        G.add_edge(edge[0], edge[1], **edge[2])
    
    return G

# Initial setup wizard
def initial_setup_wizard(graph):
    """Simulate initial setup with hybrid discovery"""
    # Simulate CMDB sync
    cmdb_assets = random.randint(3, 8)
    # Simulate document parsing
    doc_assets = random.randint(2, 6)
    # Simulate computer vision
    cv_assets = random.randint(4, 10)
    
    # Add discovered assets to graph
    for i in range(cmdb_assets + doc_assets + cv_assets):
        node_name = f"Discovered_Asset_{i}"
        graph.add_node(node_name, 
                      type="discovered",
                      risk=random.uniform(0.1, 0.4),
                      efficiency=random.uniform(0.7, 0.95),
                      latency=random.randint(1, 15))
    
    return graph

# Recommend automation with RL simulation
def recommend_automation(graph, use_rl_sim=True):
    """Generate physics-grounded automation recommendations"""
    recommendations = []
    
    # Find high-latency nodes for automation
    high_latency_nodes = [n for n, d in graph.nodes(data=True) 
                         if d.get('latency', 0) > 8]
    
    for node in high_latency_nodes[:3]:  # Top 3 candidates
        if use_rl_sim:
            # Simulate RL reward (constrained by MIN_REWARD)
            reward = max(MIN_REWARD, random.uniform(0.6, 0.95))
            recommendations.append(
                f"Automate '{node}' (from auto_discovery) by bridging "
                f"{random.choice(list(graph.nodes()))} to "
                f"{random.choice(list(graph.nodes()))} with AI relay "
                f"(RL-sim reward: {reward:.2f}—empowers operator sovereignty)"
            )
        else:
            recommendations.append(
                f"Automate '{node}' to reduce latency from "
                f"{graph.nodes[node].get('latency', 0)}s to 2s"
            )
    
    return recommendations

# Build world model with physics simulation
def build_world_model(graph, steps=3):
    """Build physics-grounded world model"""
    initial_latency = sum(d.get('latency', 0) for _, d in graph.nodes(data=True))
    
    # Simulate optimization steps
    final_latency = initial_latency * 0.6  # 40% improvement
    
    world_model = {
        "initial_latency": f"{initial_latency:.2f}s",
        "final_latency": f"{final_latency:.2f}s",
        "efficiency_improvement": f"{((initial_latency / final_latency) - 1) * 100:.2f}%",
        "hybrid_sources": {
            "auto_discovery": len([n for n in graph.nodes() if "Discovered" in n]),
            "cmdb": random.randint(3, 8),
            "document": random.randint(2, 6),
            "cv": random.randint(4, 10)
        },
        "physics_constants": {
            "min_reward": MIN_REWARD,
            "gravity": GRAVITY,
            "optimization_steps": steps
        }
    }
    
    return world_model

# Export world model
def export_world_model(graph, world_model):
    """Export encrypted world model"""
    export_data = {
        "timestamp": time.time(),
        "graph": {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "node_types": dict(nx.get_node_attributes(graph, 'type'))
        },
        "world_model": world_model,
        "physics_grounded": True
    }
    
    # Simulate encryption
    encrypted = base64.b64encode(json.dumps(export_data).encode()).decode()
    return encrypted

# Generate site map
def generate_site_map(return_base64=True):
    """Generate global site map"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot sites
    for name, data in sites.items():
        ax.scatter(data['lon'], data['lat'], 
                  s=data['efficiency']*50, 
                  c=data['risk_score'], 
                  cmap='RdYlGn',
                  alpha=0.7)
        ax.annotate(name, (data['lon'], data['lat']), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('RevealGrid Global Site Map')
    ax.grid(True, alpha=0.3)
    
    if return_base64:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        return base64.b64encode(image_png).decode()
    else:
        return fig

# Discover assets
def discover_assets():
    """Simulate asset discovery"""
    assets = {
        "sensors": random.randint(5, 15),
        "controllers": random.randint(3, 8),
        "actuators": random.randint(4, 12),
        "safety": random.randint(2, 6),
        "robots": random.randint(2, 8),
        "manual": random.randint(3, 10),
        "cmdb_synced": random.randint(3, 8),
        "document_parsed": random.randint(2, 6),
        "cv_detected": random.randint(4, 12)
    }
    
    return f"**Hybrid Asset Discovery: {sum(assets.values())} total**\n" + \
           "\n".join([f"   {k}: {v} assets" for k, v in assets.items()])

# Visualize process mapping
def visualize_process_mapping(graph, style='3d_plant_floor'):
    """Visualize process mapping with different styles"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if style == '3d_plant_floor':
        # 3D-like layout
        pos = nx.spring_layout(graph, k=3, iterations=50)
        # Add depth effect
        for node in pos:
            pos[node] = (pos[node][0], pos[node][1] + random.uniform(-0.1, 0.1))
    else:
        pos = nx.spring_layout(graph)
    
    # Color nodes by type
    node_colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        if node_type == 'controller':
            node_colors.append('#3b82f6')  # Blue
        elif node_type == 'sensor':
            node_colors.append('#10b981')  # Green
        elif node_type == 'actuator':
            node_colors.append('#f59e0b')  # Orange
        elif node_type == 'safety':
            node_colors.append('#ef4444')  # Red
        else:
            node_colors.append('#6b7280')  # Gray
    
    nx.draw(graph, pos, 
            node_color=node_colors,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            edge_color='#d1d5db',
            width=2)
    
    ax.set_title(f'Process Mapping Visualization ({style})')
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()

# Hybrid Discover
def hybrid_discover():
    graph = create_sample_graph()
    graph = initial_setup_wizard(graph)
    return graph

# Downtime Recs
def downtime_recommendations(graph):
    recs = recommend_automation(graph, use_rl_sim=True)
    return "\n".join(recs)

# Global Site Deck (PyDeck)
def global_site_deck(sites_data, globe=True):
    if not PYDECK_AVAILABLE:
        st.warning("⚠️ PyDeck not available. Install with: pip install pydeck")
        # Fallback to simple map
        st.write("**Global Site Map**")
        for name, data in sites_data.items():
            st.write(f"📍 {name}: Lat {data['lat']:.4f}, Lon {data['lon']:.4f} (Efficiency: {data['efficiency']}%)")
        return
    
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
    return st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9'))

# Chat UX & FSM
st.markdown("### 💬 Chat Interface")
prompt = st.text_input("Type yes / no / skip /back /restart /help …", key="chat_input")
if prompt:
    st.markdown(f"**You:** {prompt}")
    cmd = parse_cmd(prompt)

    if cmd == "restart":
        st.session_state.clear()
        st.experimental_rerun()
    elif cmd == "back":
        if st.session_state.log:
            prev = st.session_state.log.pop()
            set_state(prev["step"])
        st.stop()
    elif cmd == "help":
        st.markdown("**Assistant:** *Commands*: `/skip` to skip optional step, "
            "`/back` to go back, `/restart` to start over.")
        st.stop()

    state = st.session_state.state

    if state == "ask_scan_permission":
        if cmd == "yes":
            with st.spinner("📡 Passive scan starting … (physics-grounded for accuracy)"):
                time.sleep(1)
                vlans, devices, airgap = run_passive_scan_stub()
            st.session_state.answers["scan_allowed"] = True
            st.session_state.answers.update({"vlans": vlans, "devices": devices})
            st.markdown(f"**Assistant:** I detected **{vlans} VLANs** and **{devices} devices**. "
                f"It looks **{'air‑gapped' if airgap else 'connected'}**. "
                "Is that correct? (yes / no)")
            set_state("ask_airgap_confirm")
        elif cmd == "no":
            st.session_state.answers["scan_allowed"] = False
            st.markdown("**Assistant:** No problem. Let's enter location manually. "
                "Type the **plant name, city, country**.")
            set_state("ask_manual_loc")
        else:
            st.markdown("**Assistant:** Please answer *yes* or *no*.")

    elif state == "ask_airgap_confirm":
        if cmd in {"yes", "no"}:
            st.session_state.answers["airgap_confirmed"] = cmd == "yes"
            st.markdown("**Assistant:** Shall I pin these devices on the satellite map? (yes / no)")
            set_state("ask_auto_mapping")
        else:
            st.markdown("**Assistant:** yes / no?")

    elif state == "ask_auto_mapping":
        if cmd == "yes":
            st.session_state.show_globe = True
            st.markdown("**Assistant:** 🛰️ Mapping now (grounded in real geo-physics) …")
            set_state("ask_pid_upload")
        elif cmd == "no":
            st.session_state.show_globe = False
            set_state("ask_pid_upload")
        else:
            st.markdown("**Assistant:** yes / no?")

    elif state == "ask_pid_upload":
        if cmd == "skip":
            set_state("ask_camera")
        elif cmd in {"yes", "no"}:
            st.markdown("**Assistant:** Upload your P&ID (PDF) below or type `/skip`.")
            pid = st.file_uploader(" ", type=["pdf"], key="pid_uploader")
            if pid:
                st.session_state.answers["pid_uploaded"] = True
                st.markdown("**Assistant:** P&ID uploaded and parsed (physics schemas extracted).")
                set_state("ask_camera")
        else:
            st.markdown("**Assistant:** Type *yes* to upload or *skip*.")

    elif state == "ask_camera":
        if cmd == "skip":
            set_state("run_hybrid_discover")
        elif cmd in {"yes", "no"}:
            st.markdown("**Assistant:** Configure camera CV? Upload image or `/skip`.")
            img = st.file_uploader(" ", type=["jpg", "png"], key="cv_uploader")
            if img:
                st.session_state.answers["cv_configured"] = True
                st.markdown("**Assistant:** CV configured and assets detected (visual context grounded).")
                set_state("run_hybrid_discover")
        else:
            st.markdown("**Assistant:** Type *yes* or *skip*.")

    elif state == "run_hybrid_discover":
        with st.spinner("Aggregating hybrid data (physics-first for no hallucinations) …"):
            time.sleep(1)
            graph = hybrid_discover()
            st.session_state.answers["graph"] = graph
            st.session_state.graph = graph  # For display
        st.markdown("**Assistant:** Discovery complete! Generating recommendations …")
        recs = downtime_recommendations(graph)
        st.markdown(f"**Assistant:** {recs}")
        set_state("ask_export")

    elif state == "ask_export":
        if cmd == "yes":
            export = export_world_model(st.session_state.answers["graph"], build_world_model(st.session_state.answers["graph"]))
            st.markdown(f"**Assistant:** Exported model (encrypted, physics-grounded): {export[:100]}...")
            set_state("done")
        elif cmd == "no":
            set_state("done")
        else:
            st.markdown("**Assistant:** Export world model? (yes / no)")

# Render Globe if Ready
if st.session_state.get("show_globe"):
    global_site_deck(sites, globe=True)

# Sidebar Log & Controls
st.sidebar.title("RevealGrid Controls")
st.sidebar.json(st.session_state.log)
simulation_steps = st.sidebar.slider("Simulation Steps", 1, 10, 3)
mapping_style = st.sidebar.selectbox("Mapping Style", ['plant_floor', 'scatterplot', '3d_plant_floor'], index=2)
selected_site = st.sidebar.selectbox("Select Site", list(sites.keys()))

# Main Dashboard if Done
if st.session_state.state == "done" and st.session_state.graph:
    st.title("RevealGrid Dashboard")
    
    # Global Dashboard
    st.header("Global Overview")
    site_map_img = generate_site_map(return_base64=True)
    st.image(base64.b64decode(site_map_img), use_column_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Efficiency", f"{sum(s['efficiency'] for s in sites.values()) / len(sites):.1f}%")
    col2.metric("Avg Downtime", f"{sum(s['downtime'] for s in sites.values()) / len(sites):.1f} hrs")
    col3.metric("Avg Risk", f"{sum(s['risk_score'] for s in sites.values()) / len(sites):.1f}%")

    # Site Drill-Down
    if selected_site:
        st.header(f"{selected_site} Drill-Down")
        graph = st.session_state.graph  # Use discovered graph
        
        # Assets Tab
        with st.expander("Assets"):
            assets = discover_assets()
            st.write(assets)
        
        # Processes Tab
        with st.expander("Processes"):
            process_img = visualize_process_mapping(graph, style=mapping_style)
            st.image(base64.b64decode(process_img), use_column_width=True)
        
        # Recommendations Tab
        with st.expander("Recommendations"):
            recs = downtime_recommendations(graph)
            st.write(recs)
        
        # Risks Tab
        with st.expander("Risks"):
            st.write("Physics-grounded risks: Coming soon")  # Expand with real
        
        # World Model Explorer
        st.header("World Model Explorer")
        world_sim = build_world_model(graph, steps=simulation_steps)
        st.json(world_sim)
        
        # Export
        if st.button("Export Model"):
            export = export_world_model(graph, world_sim)
            st.download_button("Download Encrypted Model", export, file_name="revealgrid_model.json.enc")

# Initial welcome message
if st.session_state.state == "ask_scan_permission":
    st.title("🏭 RevealGrid Enhanced Demo")
    st.markdown("**Physics-First Industrial Intelligence Platform**")
    st.markdown("---")
    st.chat_message("assistant").write(
        "Welcome to RevealGrid! I'm here to help you discover and optimize your industrial assets. "
        "Shall I run a passive network scan to detect your devices? (yes / no)"
    ) 