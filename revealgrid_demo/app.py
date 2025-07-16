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
import torch
import torch.nn as nn
from revealgrid_core import (
    global_site_map, sample_graph, plot_process,
    hybrid_discover, downtime_recommendations
)

# ---------- 0. Init ----------
if "state" not in st.session_state:
    st.session_state.state = "start"
    st.session_state.sites = {}
    st.session_state.graph = sample_graph()
    st.session_state.log = []
    st.session_state.scan_allowed = False
    st.session_state.env_type = None
    st.session_state.skip_auto_map = False
    st.session_state.current_site = None
    st.session_state.answers = {}
    st.session_state.show_globe = False

# Custom CSS for professional layout
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    .risk-high { border-left: 4px solid #ef4444; }
    .risk-medium { border-left: 4px solid #f59e0b; }
    .risk-low { border-left: 4px solid #10b981; }
</style>
""", unsafe_allow_html=True)

# Centered header
st.markdown('<h1 class="main-header">RevealGrid</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Industrial Asset Intelligence Platform</p>', unsafe_allow_html=True)

# ---------- 1. Enhanced command parser ----------
def parse_cmd(txt: str):
    """
    Returns one of: 'yes', 'no', 'skip', 'back', 'restart', 'help', None
    """
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

def set_state(new_state):
    """Atomic state transition"""
    st.session_state.state = new_state

def log_step(step, answer):
    st.session_state.log.append({
        "step": step, 
        "answer": answer, 
        "timestamp": time.time()
    })

def bot(msg):
    st.write(f"**RevealGrid:** {msg}")

def user_msg(msg):
    st.write(f"**You:** {msg}")

# ---------- 2. Stub functions for RevealGrid integration ----------
def run_passive_scan_stub():
    """Simulate passive network discovery"""
    time.sleep(1)  # Simulate scan time
    return 3, 214, True  # vlans, devices, airgap

def run_hybrid_discover_stub():
    """Simulate hybrid asset discovery"""
    time.sleep(2)  # Simulate discovery time
    return sample_graph()

def run_downtime_recommendations_stub(graph):
    """Simulate downtime recommendations"""
    time.sleep(1)  # Simulate analysis time
    return [
        "Automate manual valve operations to reduce downtime by 40%",
        "Implement predictive maintenance for critical pumps",
        "Optimize conveyor belt scheduling for 25% efficiency gain"
    ]

def export_world_model_stub(graph):
    """Simulate world model export"""
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "encrypted_data": "eyJ0aW1lc3RhbXAiOiAiMjAyNS0wNy0xMVQxMzowMDowMC4wMDAwMDAiLCJkYXRhIjogImVuY3J5cHRlZCJ9"
    }

# ---------- 3. Display conversation history ----------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for entry in st.session_state.log:
    if entry["step"] == "user":
        user_msg(entry["answer"])
    else:
        bot(entry["answer"])

# ---------- 4. Show current question if no input yet ----------
current_state = st.session_state.state

if current_state == "start":
    bot("Welcome to RevealGrid. I can automatically discover and map your industrial assets by running a read-only passive scan on this subnet (no packets sent, just listening). This will help us build a comprehensive world model of your OT environment. May I begin? (yes / no / skip)")

elif current_state == "ask_airgap_confirm":
    bot("I detected 3 VLANs, 214 OT devices including PLCs, sensors, and actuators. The network appears air-gapped (no internet gateway). This helps us understand your security posture. Is that correct? (yes / no / skip)")

elif current_state == "ask_auto_mapping":
    bot("Great. Now I can create a visual representation of your assets on a global map. This helps with fleet management and operational visibility across multiple sites. Shall I pin these devices on the global satellite map using registered plant locations? (yes / no / skip)")

elif current_state == "ask_manual_loc":
    bot("What's the plant name and location? This helps us geolocate your assets for fleet management. (e.g., 'Detroit Engine Plant, Detroit USA' / skip)")

elif current_state == "ask_manual_network":
    bot("What's the network scope? This helps us understand the boundaries of your OT environment for asset discovery. (e.g., '192.168.1.0/24' or 'VLAN 10' / skip)")

elif current_state == "ask_pid_upload":
    bot("Would you like to upload a P&ID (Piping & Instrumentation Diagram) or PDF? This helps us identify offline assets like valves, pumps, and manual processes that aren't connected to the network. (yes / no / skip)")

elif current_state == "ask_camera_feeds":
    bot("Any camera feeds I should include for visual context? This enables computer vision for asset monitoring and autonomous operations. (yes / no / skip)")

elif current_state == "ask_compliance":
    bot("Would you like baseline compliance checks (NIST/ISO)? This helps assess your security posture and identify potential vulnerabilities. (yes / no / skip)")

elif current_state == "run_hybrid_discover":
    bot("Running hybrid asset discovery... This combines network scans, document parsing, and computer vision to build a comprehensive asset inventory.")

elif current_state == "show_recommendations":
    bot("Discovery complete! Here are your downtime and risk recommendations:")

elif current_state == "ask_export":
    bot("Would you like to export your world model for secure storage or sharing? This creates an encrypted snapshot of your asset inventory. (yes / no / skip)")

elif current_state == "ready":
    bot("Setup complete! Your industrial asset world model is ready. Type 'insight' to get detailed analysis, 'add' to map another site, 'export' for a secure model, or 'help'.")

# ---------- 5. Chat input and command processing ----------
prompt = st.text_input("Your response:", key="chat_input")
if prompt and st.button("Send", key="send_btn"):
    # Log user input immediately
    log_step("user", prompt)
    
    # Parse command
    cmd = parse_cmd(prompt)
    
    # ------------ command routing ------------
    if cmd == "restart":
        st.session_state.clear()
        st.session_state.state = "start"
        st.experimental_rerun()
    elif cmd == "back":
        # pop last log and rewind one step
        if st.session_state.log:
            st.session_state.log.pop()  # Remove user input
            if st.session_state.log:
                last_step = st.session_state.log[-1]["step"]
                set_state(last_step)
        st.experimental_rerun()
    elif cmd == "help":
        log_step("system", "Commands: /skip to skip optional step, /back to go back, /restart to start over")
        st.experimental_rerun()
    
    # ------------ FSM state machine ------------
    state = st.session_state.state
    
    if state == "start":
        if cmd == "yes":
            log_step("system", "Starting passive network discovery...")
            set_state("run_passive_scan")
        elif cmd == "no":
            log_step("system", "Skipping automatic discovery. Let's add sites manually.")
            set_state("ask_manual_loc")
        else:
            log_step("system", "Please answer 'yes' or 'no'.")
        st.experimental_rerun()
    
    elif state == "run_passive_scan":
        # This state auto-advances after simulation
        with st.spinner("Passive network discovery running... 214 devices found"):
            vlans, devices, airgap = run_passive_scan_stub()
            st.session_state.answers.update({
                "vlans": vlans,
                "devices": devices,
                "airgap": airgap
            })
        log_step("system", f"I detected {vlans} VLANs, {devices} OT devices including PLCs, sensors, and actuators. The network appears {'air-gapped' if airgap else 'connected'} (no internet gateway). This helps us understand your security posture. Is that correct? (yes / no / skip)")
        set_state("ask_airgap_confirm")
        st.experimental_rerun()
    
    elif state == "ask_airgap_confirm":
        if cmd == "yes":
            st.session_state.env_type = "air_gapped"
            log_step("system", "Environment classified as air-gapped OT network")
            set_state("ask_auto_mapping")
        elif cmd in ["no", "skip"]:
            st.session_state.env_type = "mixed"
            log_step("system", "Environment classified as mixed IT/OT network")
            set_state("ask_auto_mapping")
        else:
            log_step("system", "Please answer 'yes', 'no', or 'skip'.")
        st.experimental_rerun()
    
    elif state == "ask_auto_mapping":
        if cmd == "yes":
            st.session_state.sites["Detroit Plant"] = {"lat": 42.33, "lon": -83.04, "eff": 85, "down": 5, "risk": 20}
            st.session_state.show_globe = True
            log_step("system", "Global mapping enabled")
            set_state("ask_pid_upload")
        elif cmd in ["no", "skip"]:
            st.session_state.show_globe = False
            log_step("system", "Global mapping skipped")
            set_state("ask_pid_upload")
        else:
            log_step("system", "Please answer 'yes', 'no', or 'skip'.")
        st.experimental_rerun()
    
    elif state == "ask_manual_loc":
        if cmd == "skip":
            set_state("ask_manual_network")
        else:
            st.session_state.answers["manual_location"] = prompt
            log_step("system", f"Location recorded: {prompt}")
            set_state("ask_manual_network")
        st.experimental_rerun()
    
    elif state == "ask_manual_network":
        if cmd == "skip":
            set_state("ask_pid_upload")
        else:
            st.session_state.answers["manual_network"] = prompt
            log_step("system", f"Network scope recorded: {prompt}")
            set_state("ask_pid_upload")
        st.experimental_rerun()
    
    elif state == "ask_pid_upload":
        if cmd == "skip":
            log_step("system", "P&ID upload skipped")
            set_state("ask_camera_feeds")
        elif cmd == "yes":
            log_step("system", "Please upload your P&ID document below:")
            # File uploader would go here
            st.session_state.answers["pid_uploaded"] = True
            set_state("ask_camera_feeds")
        else:
            log_step("system", "Please answer 'yes' or 'skip'.")
        st.experimental_rerun()
    
    elif state == "ask_camera_feeds":
        if cmd == "skip":
            log_step("system", "Camera feeds skipped")
            set_state("ask_compliance")
        elif cmd == "yes":
            log_step("system", "Please upload camera feed images below:")
            # File uploader would go here
            st.session_state.answers["camera_feeds"] = True
            set_state("ask_compliance")
        else:
            log_step("system", "Please answer 'yes' or 'skip'.")
        st.experimental_rerun()
    
    elif state == "ask_compliance":
        if cmd == "skip":
            log_step("system", "Compliance checks skipped")
            set_state("run_hybrid_discover")
        elif cmd == "yes":
            log_step("system", "Running compliance checks (NIST/ISO)...")
            st.session_state.answers["compliance_checked"] = True
            set_state("run_hybrid_discover")
        else:
            log_step("system", "Please answer 'yes' or 'skip'.")
        st.experimental_rerun()
    
    elif state == "run_hybrid_discover":
        with st.spinner("Running hybrid asset discovery..."):
            graph = run_hybrid_discover_stub()
            st.session_state.graph = graph
            st.session_state.answers["discovered_assets"] = len(graph.nodes)
        log_step("system", f"Discovery complete! Found {len(graph.nodes)} assets across multiple sources.")
        set_state("show_recommendations")
        st.experimental_rerun()
    
    elif state == "show_recommendations":
        recommendations = run_downtime_recommendations_stub(st.session_state.graph)
        log_step("system", "Downtime and risk recommendations:")
        for rec in recommendations:
            log_step("system", f"• {rec}")
        set_state("ask_export")
        st.experimental_rerun()
    
    elif state == "ask_export":
        if cmd == "yes":
            export_data = export_world_model_stub(st.session_state.graph)
            log_step("system", f"World model exported successfully! {export_data['nodes']} nodes, {export_data['edges']} edges")
            set_state("ready")
        elif cmd in ["no", "skip"]:
            log_step("system", "Export skipped")
            set_state("ready")
        else:
            log_step("system", "Please answer 'yes', 'no', or 'skip'.")
        st.experimental_rerun()

# ---------- 6. Render additional UI elements ----------
if st.session_state.get("show_globe"):
    st.subheader("🌍 Global Site Map")
    st.write("Your assets have been mapped to the global satellite view.")
    # Placeholder for PyDeck chart
    st.info("Global mapping visualization would appear here")

# Progress indicator
if st.session_state.log:
    st.sidebar.subheader("📊 Setup Progress")
    progress = len([e for e in st.session_state.log if e["step"] != "user"]) / 10 * 100
    st.sidebar.progress(min(progress, 100))
    st.sidebar.write(f"Steps completed: {len([e for e in st.session_state.log if e['step'] != 'user'])}")

# Debug log (collapsible)
with st.sidebar.expander("🔧 Debug Log"):
    st.json(st.session_state.log)

st.markdown('</div>', unsafe_allow_html=True) 