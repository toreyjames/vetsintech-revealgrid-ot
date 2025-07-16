import streamlit as st
import time
import json
import networkx as nx
import random
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Set page config
st.set_page_config(
    page_title="RevealGrid - Industrial Asset Intelligence",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for professional look
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

# Initialize session state
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0
    st.session_state.assets_discovered = 0
    st.session_state.recommendations = []
    st.session_state.risk_insights = []

# Header
st.markdown('<h1 class="main-header">RevealGrid</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Industrial Asset Intelligence Platform</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏭 RevealGrid Demo")
st.sidebar.markdown("**VetinTech Event Demo**")

# Demo steps
steps = [
    "Welcome & Overview",
    "Asset Discovery",
    "Risk Analysis", 
    "Automation Recommendations",
    "World Model Export"
]

# Progress bar
progress = (st.session_state.demo_step + 1) / len(steps)
st.sidebar.progress(progress)
st.sidebar.write(f"Step {st.session_state.demo_step + 1} of {len(steps)}")

# Main content based on demo step
if st.session_state.demo_step == 0:
    st.header("🎯 Welcome to RevealGrid")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **RevealGrid** is an AI-driven industrial intelligence platform that:
        
        🔍 **Discovers** industrial assets automatically
        ⚠️ **Analyzes** risks and vulnerabilities  
        🤖 **Recommends** automation opportunities
        🌍 **Simulates** world models for optimization
        📦 **Exports** secure asset inventories
        
        Perfect for OT environments requiring:
        - Air-gapped security compliance
        - Hybrid asset discovery (network + documents + CV)
        - Predictive maintenance insights
        - Robot integration planning
        """)
    
    with col2:
        st.markdown("""
        **Demo Features:**
        
        📊 Real-time asset discovery simulation
        🔍 Risk assessment with severity levels
        ⚡ AI-powered automation recommendations
        🌍 World model simulation & export
        📈 Performance optimization metrics
        
        **Use Cases:**
        - Manufacturing plants
        - Power generation facilities
        - Oil & gas operations
        - Water treatment systems
        - Smart city infrastructure
        """)
    
    if st.button("🚀 Start Demo", type="primary"):
        st.session_state.demo_step = 1
        st.experimental_rerun()

elif st.session_state.demo_step == 1:
    st.header("🔍 Asset Discovery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Discovery Methods")
        
        # Simulate discovery process
        if st.button("🔍 Start Discovery"):
            with st.spinner("Discovering industrial assets..."):
                time.sleep(2)
                st.session_state.assets_discovered = random.randint(15, 25)
                st.success(f"✅ Discovered {st.session_state.assets_discovered} assets!")
        
        if st.session_state.assets_discovered > 0:
            st.markdown(f"""
            **Discovery Results:**
            - **Network Scan:** {st.session_state.assets_discovered // 3} devices
            - **Document Parsing:** {st.session_state.assets_discovered // 4} assets
            - **Computer Vision:** {st.session_state.assets_discovered // 5} visual assets
            - **CMDB Sync:** {st.session_state.assets_discovered // 6} records
            
            **Asset Types Found:**
            - Sensors: {st.session_state.assets_discovered // 4}
            - Controllers: {st.session_state.assets_discovered // 6}
            - Actuators: {st.session_state.assets_discovered // 5}
            - Safety Systems: {st.session_state.assets_discovered // 7}
            """)
    
    with col2:
        st.subheader("Asset Map")
        
        # Create a simple network visualization
        if st.session_state.assets_discovered > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create a simple graph
            G = nx.random_geometric_graph(st.session_state.assets_discovered, 0.3)
            pos = nx.spring_layout(G)
            
            nx.draw(G, pos, ax=ax, 
                   node_color='lightblue',
                   node_size=100,
                   with_labels=False)
            
            ax.set_title(f"Asset Network ({st.session_state.assets_discovered} nodes)")
            st.pyplot(fig)
    
    if st.session_state.assets_discovered > 0:
        if st.button("⚠️ Analyze Risks"):
            st.session_state.demo_step = 2
            st.experimental_rerun()

elif st.session_state.demo_step == 2:
    st.header("⚠️ Risk Analysis")
    
    # Generate risk insights
    if not st.session_state.risk_insights:
        risk_types = ["High Latency", "Failure Probability", "Security Vulnerability", "Bottleneck"]
        assets = ["TemperatureSensor", "PressureSensor", "PLC_Controller", "Actuator_Valve", 
                 "SafetySystem", "DataLogger", "NetworkSwitch", "RobotArm"]
        
        st.session_state.risk_insights = []
        for _ in range(random.randint(8, 12)):
            risk = {
                "asset": random.choice(assets),
                "type": random.choice(risk_types),
                "severity": random.choice(["high", "medium", "low"]),
                "probability": random.uniform(15, 35),
                "description": f"Potential {random.choice(risk_types).lower()} detected"
            }
            st.session_state.risk_insights.append(risk)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Overview")
        
        high_risks = len([r for r in st.session_state.risk_insights if r["severity"] == "high"])
        medium_risks = len([r for r in st.session_state.risk_insights if r["severity"] == "medium"])
        low_risks = len([r for r in st.session_state.risk_insights if r["severity"] == "low"])
        
        st.metric("High Severity Risks", high_risks, delta=f"+{high_risks}")
        st.metric("Medium Severity Risks", medium_risks, delta=f"+{medium_risks}")
        st.metric("Low Severity Risks", low_risks, delta=f"+{low_risks}")
        
        # Risk distribution chart
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['High', 'Medium', 'Low']
        sizes = [high_risks, medium_risks, low_risks]
        colors = ['#ef4444', '#f59e0b', '#10b981']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%')
        ax.set_title("Risk Distribution by Severity")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Detailed Risk Analysis")
        
        for risk in st.session_state.risk_insights[:6]:  # Show first 6
            severity_color = {
                "high": "#ef4444",
                "medium": "#f59e0b", 
                "low": "#10b981"
            }
            
            st.markdown(f"""
            <div class="metric-card risk-{risk['severity']}">
                <strong>{risk['asset']}</strong><br>
                <strong>Type:</strong> {risk['type']}<br>
                <strong>Severity:</strong> {risk['severity'].title()}<br>
                <strong>Probability:</strong> {risk['probability']:.1f}%<br>
                <strong>Description:</strong> {risk['description']}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("⚡ Generate Recommendations"):
        st.session_state.demo_step = 3
        st.experimental_rerun()

elif st.session_state.demo_step == 3:
    st.header("⚡ Automation Recommendations")
    
    # Generate recommendations
    if not st.session_state.recommendations:
        actions = ["Automate", "Optimize", "Implement", "Deploy"]
        targets = ["ManualCheck_Operator", "Actuator_Valve", "BottlingMachine", "QualityControl_Station"]
        benefits = ["reduce downtime by 40%", "improve efficiency by 25%", "increase safety by 60%", "optimize throughput by 35%"]
        
        st.session_state.recommendations = []
        for _ in range(random.randint(4, 6)):
            rec = {
                "action": random.choice(actions),
                "target": random.choice(targets),
                "benefit": random.choice(benefits),
                "priority": random.choice(["high", "medium"]),
                "reward": random.uniform(0.7, 0.95)
            }
            st.session_state.recommendations.append(rec)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recommendation Summary")
        
        high_priority = len([r for r in st.session_state.recommendations if r["priority"] == "high"])
        total_recs = len(st.session_state.recommendations)
        avg_reward = sum(r["reward"] for r in st.session_state.recommendations) / len(st.session_state.recommendations)
        
        st.metric("Total Recommendations", total_recs)
        st.metric("High Priority Actions", high_priority)
        st.metric("Average AI Reward", f"{avg_reward:.2f}")
        
        # Reward distribution
        rewards = [r["reward"] for r in st.session_state.recommendations]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rewards, bins=5, color='lightblue', edgecolor='black')
        ax.set_xlabel("AI Reward Score")
        ax.set_ylabel("Number of Recommendations")
        ax.set_title("Recommendation Quality Distribution")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Detailed Recommendations")
        
        for i, rec in enumerate(st.session_state.recommendations, 1):
            priority_color = "#ef4444" if rec["priority"] == "high" else "#f59e0b"
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Recommendation {i}</strong><br>
                <strong>Action:</strong> {rec['action']} {rec['target']}<br>
                <strong>Benefit:</strong> {rec['benefit']}<br>
                <strong>Priority:</strong> <span style="color: {priority_color}">{rec['priority'].title()}</span><br>
                <strong>AI Reward:</strong> {rec['reward']:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🌍 Build World Model"):
        st.session_state.demo_step = 4
        st.experimental_rerun()

elif st.session_state.demo_step == 4:
    st.header("🌍 World Model Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Statistics")
        
        # Generate model stats
        nodes = st.session_state.assets_discovered
        edges = nodes * random.uniform(0.8, 1.2)
        efficiency = random.uniform(85, 95)
        latency_improvement = random.uniform(30, 60)
        
        st.metric("Total Nodes", nodes)
        st.metric("Total Edges", int(edges))
        st.metric("System Efficiency", f"{efficiency:.1f}%")
        st.metric("Latency Improvement", f"{latency_improvement:.1f}%")
        
        # Performance chart
        steps = list(range(1, 11))
        latencies = [100 - (i * latency_improvement / 10) for i in range(10)]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, latencies, marker='o', color='blue')
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("System Latency (s)")
        ax.set_title("World Model Optimization Progress")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Export Options")
        
        st.markdown("""
        **Export Formats:**
        - 🔐 Encrypted JSON (recommended)
        - 📊 CSV Report
        - 🗺️ GraphML Network
        - 📋 PDF Summary
        """)
        
        # Simulate export
        if st.button("🔐 Export Encrypted Model"):
            with st.spinner("Generating encrypted export..."):
                time.sleep(2)
                
                # Create mock export data
                export_data = {
                    "timestamp": time.time(),
                    "nodes": nodes,
                    "edges": int(edges),
                    "efficiency": efficiency,
                    "risks_identified": len(st.session_state.risk_insights),
                    "recommendations": len(st.session_state.recommendations)
                }
                
                # Simulate encryption
                encrypted = base64.b64encode(json.dumps(export_data).encode()).decode()
                
                st.success("✅ Export generated successfully!")
                st.code(encrypted[:100] + "...", language="text")
                
                # Download button
                st.download_button(
                    label="📥 Download Encrypted Model",
                    data=encrypted,
                    file_name="revealgrid_world_model.enc",
                    mime="text/plain"
                )
    
    st.markdown("---")
    st.markdown("""
    ### 🎉 Demo Complete!
    
    **RevealGrid** has successfully:
    - ✅ Discovered industrial assets
    - ✅ Analyzed risks and vulnerabilities
    - ✅ Generated AI-powered recommendations
    - ✅ Built and exported world model
    
    **Ready for production deployment in OT environments!**
    """)
    
    if st.button("🔄 Restart Demo"):
        st.session_state.demo_step = 0
        st.session_state.assets_discovered = 0
        st.session_state.recommendations = []
        st.session_state.risk_insights = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
    <strong>RevealGrid</strong> - Industrial Asset Intelligence Platform<br>
    Built for OT environments • Air-gapped compliant • AI-powered insights
</div>
""", unsafe_allow_html=True) 