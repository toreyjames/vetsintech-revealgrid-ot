import streamlit as st
import time
import json
import networkx as nx
import random
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from datetime import datetime

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .step-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .discovery-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .integration-card {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .process-card {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .world-model-card {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'discovered_assets' not in st.session_state:
    st.session_state.discovered_assets = []
if 'risk_insights' not in st.session_state:
    st.session_state.risk_insights = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'integrations' not in st.session_state:
    st.session_state.integrations = []
if 'process_flows' not in st.session_state:
    st.session_state.process_flows = []
if 'world_models' not in st.session_state:
    st.session_state.world_models = []

# Simulated data
sites = {
    'Plant A - Detroit': {'lat': 42.3314, 'lon': -83.0458, 'efficiency': 85, 'downtime': 5, 'risk_score': 20},
    'Plant B - Chicago': {'lat': 41.8781, 'lon': -87.6298, 'efficiency': 92, 'downtime': 2, 'risk_score': 10},
    'Plant C - Austin': {'lat': 30.2672, 'lon': -97.7431, 'efficiency': 78, 'downtime': 10, 'risk_score': 35},
    'Plant D - Seattle': {'lat': 47.6062, 'lon': -122.3321, 'efficiency': 88, 'downtime': 4, 'risk_score': 15}
}

def create_sample_graph():
    """Create a sample industrial network graph"""
    G = nx.Graph()
    
    # Add industrial assets
    assets = [
        ('TemperatureSensor', 'sensor'),
        ('PressureSensor', 'sensor'),
        ('FlowMeter', 'sensor'),
        ('PLC_Controller', 'controller'),
        ('SCADASystem', 'controller'),
        ('Actuator_Valve', 'actuator'),
        ('Actuator_Pump', 'actuator'),
        ('BottlingMachine', 'machine'),
        ('QualityControl_Station', 'machine'),
        ('SafetySystem', 'safety'),
        ('SafetyBarrier', 'safety'),
        ('DataLogger', 'data'),
        ('NetworkSwitch', 'network'),
        ('OTController', 'controller')
    ]
    
    for asset, asset_type in assets:
        G.add_node(asset, type=asset_type, latency=random.randint(5, 15))
    
    # Add connections
    for i in range(len(assets) - 1):
        G.add_edge(assets[i][0], assets[i + 1][0])
    
    return G

def generate_site_map():
    """Generate site map visualization"""
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
        ax.text(data['lon'] + 0.5, data['lat'], 
                f"{name}\nEff: {data['efficiency']}%\nDown: {data['downtime']}%\nRisk: {data['risk_score']}%", 
                fontsize=9)
    
    plt.colorbar(scatter, label='Efficiency %')
    
    # Convert to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def simulate_pid_analysis():
    """Simulate P&ID drawing analysis"""
    pid_assets = [
        'Control_Valve_CV001', 'Pressure_Transmitter_PT101', 'Flow_Transmitter_FT201',
        'Temperature_Transmitter_TT301', 'Level_Transmitter_LT401', 'Control_Valve_CV002',
        'Pressure_Transmitter_PT102', 'Flow_Transmitter_FT202', 'Temperature_Transmitter_TT302'
    ]
    return random.sample(pid_assets, random.randint(6, 9))

def simulate_drawing_analysis():
    """Simulate CAD drawing analysis"""
    cad_assets = [
        'Pump_P101', 'Compressor_C201', 'Heat_Exchanger_E301', 'Tank_T401',
        'Vessel_V501', 'Pump_P102', 'Compressor_C202', 'Heat_Exchanger_E302'
    ]
    return random.sample(cad_assets, random.randint(5, 8))

def simulate_software_integration():
    """Simulate software integrations"""
    integrations = [
        {'name': 'Siemens TIA Portal', 'assets': 12, 'status': 'Connected'},
        {'name': 'Rockwell Studio 5000', 'assets': 8, 'status': 'Connected'},
        {'name': 'Schneider Electric EcoStruxure', 'assets': 6, 'status': 'Connected'},
        {'name': 'Honeywell Experion PKS', 'assets': 4, 'status': 'Connected'},
        {'name': 'ABB 800xA', 'assets': 7, 'status': 'Connected'},
        {'name': 'Emerson DeltaV', 'assets': 5, 'status': 'Connected'}
    ]
    return integrations

def simulate_process_analysis():
    """Simulate deep learning process analysis"""
    processes = [
        {
            'name': 'Raw Material Processing',
            'steps': ['Material_Intake', 'Quality_Check', 'Preprocessing', 'Storage'],
            'efficiency': 87,
            'bottlenecks': ['Quality_Check'],
            'optimization_potential': 23
        },
        {
            'name': 'Production Line A',
            'steps': ['Assembly_Start', 'Component_Integration', 'Testing', 'Packaging'],
            'efficiency': 92,
            'bottlenecks': ['Testing'],
            'optimization_potential': 15
        },
        {
            'name': 'Quality Control',
            'steps': ['Inspection', 'Testing', 'Certification', 'Release'],
            'efficiency': 78,
            'bottlenecks': ['Certification'],
            'optimization_potential': 31
        },
        {
            'name': 'Shipping & Logistics',
            'steps': ['Order_Processing', 'Picking', 'Packing', 'Shipping'],
            'efficiency': 85,
            'bottlenecks': ['Picking'],
            'optimization_potential': 18
        }
    ]
    return processes

def generate_process_flow_chart():
    """Generate process flow visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simple process flow diagram
    processes = ['Raw Material', 'Processing', 'Assembly', 'Testing', 'Packaging', 'Shipping']
    efficiencies = [random.randint(75, 95) for _ in processes]
    
    # Create flow diagram
    y_pos = np.arange(len(processes))
    bars = ax.barh(y_pos, efficiencies, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(processes)
    ax.set_xlabel('Efficiency (%)')
    ax.set_title('AI-Derived Process Flow Analysis')
    
    # Add efficiency labels
    for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
               f'{eff}%', ha='left', va='center')
    
    # Add flow arrows
    for i in range(len(processes) - 1):
        ax.annotate('', xy=(efficiencies[i+1], i+1), xytext=(efficiencies[i], i),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    flow_chart = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return flow_chart

def simulate_world_model_unification():
    """Simulate unified world model creation"""
    world_models = [
        {
            'site': 'Plant A - Detroit',
            'assets': 156,
            'processes': 8,
            'efficiency': 85,
            'contribution': 'Automotive Manufacturing'
        },
        {
            'site': 'Plant B - Chicago',
            'assets': 203,
            'processes': 12,
            'efficiency': 92,
            'contribution': 'Chemical Processing'
        },
        {
            'site': 'Plant C - Austin',
            'assets': 89,
            'processes': 5,
            'efficiency': 78,
            'contribution': 'Electronics Assembly'
        },
        {
            'site': 'Plant D - Seattle',
            'assets': 134,
            'processes': 9,
            'efficiency': 88,
            'contribution': 'Aerospace Manufacturing'
        }
    ]
    return world_models

def simulate_asset_discovery():
    """Simulate comprehensive asset discovery process"""
    # Network discovery
    network_assets = [
        'TemperatureSensor', 'PressureSensor', 'FlowMeter', 'PLC_Controller',
        'SCADASystem', 'Actuator_Valve', 'Actuator_Pump', 'BottlingMachine'
    ]
    
    # P&ID analysis
    pid_assets = simulate_pid_analysis()
    
    # CAD drawing analysis
    cad_assets = simulate_drawing_analysis()
    
    # Software integrations
    integrations = simulate_software_integration()
    
    # Combine all discovery methods
    all_assets = network_assets + pid_assets + cad_assets
    
    return {
        'network_assets': random.sample(network_assets, random.randint(6, 8)),
        'pid_assets': pid_assets,
        'cad_assets': cad_assets,
        'integrations': integrations,
        'total_assets': len(all_assets)
    }

def simulate_risk_analysis():
    """Simulate risk analysis"""
    risks = [
        'High latency detected in Actuator_Valve (15ms)',
        'TemperatureSensor showing 25% failure probability',
        'NetworkSwitch has connectivity issues',
        'SafetySystem requires maintenance',
        'PLC_Controller showing unusual behavior patterns'
    ]
    return random.sample(risks, random.randint(3, 5))

def simulate_recommendations():
    """Simulate AI recommendations"""
    recs = [
        'Automate Actuator_Valve control for 30% efficiency gain',
        'Implement predictive maintenance for TemperatureSensor',
        'Optimize network routing through NetworkSwitch',
        'Upgrade SafetySystem firmware for improved reliability',
        'Deploy AI-powered anomaly detection for PLC_Controller'
    ]
    return random.sample(recs, random.randint(3, 5))

def export_world_model():
    """Export world model data"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'sites': len(sites),
        'total_assets': len(st.session_state.discovered_assets),
        'risk_insights': len(st.session_state.risk_insights),
        'recommendations': len(st.session_state.recommendations),
        'export_size': random.randint(50000, 150000)
    }
    return export_data

# Main app
def main():
    st.markdown('<h1 class="main-header">🏭 RevealGrid</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Industrial Asset Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Demo Progress")
    steps = ["Welcome", "Asset Discovery", "Process Analysis", "World Model", "Risk Analysis", "AI Recommendations", "Export"]
    
    for i, step in enumerate(steps):
        if i <= st.session_state.current_step:
            st.sidebar.success(f"✅ {step}")
        else:
            st.sidebar.info(f"⏳ {step}")
    
    # Main content based on current step
    if st.session_state.current_step == 0:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("🎯 Welcome to RevealGrid")
        st.write("""
        **RevealGrid** is an AI-driven industrial intelligence platform designed for OT environments. 
        Our **hybrid asset discovery** combines network scanning, P&ID analysis, CAD drawing parsing, 
        and software integrations to provide the most comprehensive industrial asset intelligence.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Global Sites", "4", "Active")
        with col2:
            st.metric("Asset Types", "8", "Monitored")
        with col3:
            st.metric("AI Models", "12", "Deployed")
        
        st.markdown("### 🔍 **Our Asset Discovery Advantage**")
        st.write("""
        Unlike competitors who only scan networks, RevealGrid provides **comprehensive asset discovery**:
        - **Network Scanning** - Real-time device discovery
        - **P&ID Analysis** - Extract assets from engineering drawings
        - **CAD Drawing Parsing** - 3D model and layout analysis
        - **Software Integrations** - Direct connections to PLC/SCADA systems
        - **Document Processing** - PDF, Excel, and technical documentation
        - **Process Analysis** - Deep learning derived flow charts
        - **Unified World Models** - Multi-site intelligence aggregation
        """)
        
        if st.button("🚀 Start Asset Discovery Demo", key="start_demo"):
            st.session_state.current_step = 1
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 1:
        st.markdown('<div class="discovery-card">', unsafe_allow_html=True)
        st.header("🔍 Advanced Asset Discovery")
        st.write("**RevealGrid's hybrid discovery engine** combines multiple data sources for comprehensive asset intelligence...")
        
        # Discovery methods
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Network Discovery")
            with st.spinner("Scanning industrial network..."):
                time.sleep(1.5)
            network_assets = ['TemperatureSensor', 'PressureSensor', 'FlowMeter', 'PLC_Controller', 'SCADASystem']
            st.success(f"✅ Discovered {len(network_assets)} network devices")
            for asset in network_assets:
                st.write(f"• {asset}")
        
        with col2:
            st.subheader("📄 P&ID Analysis")
            uploaded_file = st.file_uploader("Upload P&ID Drawing", type=['pdf', 'png', 'jpg'], key="pid_upload")
            if uploaded_file is not None:
                with st.spinner("Analyzing P&ID drawing..."):
                    time.sleep(2)
                pid_assets = simulate_pid_analysis()
                st.success(f"✅ Extracted {len(pid_assets)} assets from P&ID")
                for asset in pid_assets[:3]:  # Show first 3
                    st.write(f"• {asset}")
                if len(pid_assets) > 3:
                    st.write(f"• ... and {len(pid_assets)-3} more")
        
        # CAD Drawing Analysis
        st.subheader("🏗️ CAD Drawing Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            cad_file = st.file_uploader("Upload CAD Drawing", type=['dwg', 'dxf', 'pdf'], key="cad_upload")
            if cad_file is not None:
                with st.spinner("Parsing CAD drawing..."):
                    time.sleep(2)
                cad_assets = simulate_drawing_analysis()
                st.success(f"✅ Extracted {len(cad_assets)} assets from CAD")
        
        with col2:
            st.subheader("🔗 Software Integrations")
            integrations = simulate_software_integration()
            for integration in integrations[:3]:
                st.markdown(f"**{integration['name']}** - {integration['assets']} assets ({integration['status']})")
        
        # Comprehensive results
        st.subheader("📊 Discovery Summary")
        discovery_results = simulate_asset_discovery()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Network Assets", len(discovery_results['network_assets']))
        with col2:
            st.metric("P&ID Assets", len(discovery_results['pid_assets']))
        with col3:
            st.metric("CAD Assets", len(discovery_results['cad_assets']))
        with col4:
            st.metric("Total Assets", discovery_results['total_assets'])
        
        st.session_state.discovered_assets = discovery_results['network_assets'] + discovery_results['pid_assets'] + discovery_results['cad_assets']
        
        # Generate and display site map
        site_map = generate_site_map()
        st.image(f"data:image/png;base64,{site_map}", caption="Global Site Map with Discovered Assets")
        
        if st.button("🧠 Continue to Process Analysis", key="continue_process"):
            st.session_state.current_step = 2
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 2:
        st.markdown('<div class="process-card">', unsafe_allow_html=True)
        st.header("🧠 AI-Powered Process Analysis")
        st.write("**Deep learning and reinforcement learning** analyze discovered assets to derive high-level process flows...")
        
        with st.spinner("Analyzing process patterns with AI..."):
            time.sleep(3)
        
        st.success("✅ Process analysis complete!")
        
        # Process analysis results
        processes = simulate_process_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Discovered Processes")
            for i, process in enumerate(processes, 1):
                st.markdown(f"**{i}. {process['name']}**")
                st.write(f"Efficiency: {process['efficiency']}%")
                st.write(f"Optimization Potential: {process['optimization_potential']}%")
                st.write(f"Bottlenecks: {', '.join(process['bottlenecks'])}")
                st.write("---")
        
        with col2:
            st.subheader("🔄 Process Flow Chart")
            flow_chart = generate_process_flow_chart()
            st.image(f"data:image/png;base64,{flow_chart}", caption="AI-Derived Process Flow")
        
        st.markdown("### 🤖 **AI Analysis Insights**")
        st.write("""
        - **Deep Learning Models** analyze asset relationships and data patterns
        - **Reinforcement Learning** optimizes process flows for efficiency
        - **Pattern Recognition** identifies bottlenecks and optimization opportunities
        - **Predictive Modeling** forecasts process performance and maintenance needs
        """)
        
        if st.button("🌍 Continue to World Model", key="continue_world"):
            st.session_state.current_step = 3
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 3:
        st.markdown('<div class="world-model-card">', unsafe_allow_html=True)
        st.header("🌍 Unified World Model")
        st.write("**Each plant builds its own world model** with local data, then contributes to a unified global intelligence system...")
        
        with st.spinner("Building unified world model..."):
            time.sleep(2)
        
        st.success("✅ World model unification complete!")
        
        # World model data
        world_models = simulate_world_model_unification()
        
        st.subheader("🏭 Individual Site Models")
        col1, col2 = st.columns(2)
        
        with col1:
            for model in world_models[:2]:
                st.markdown(f"**{model['site']}**")
                st.write(f"Assets: {model['assets']}")
                st.write(f"Processes: {model['processes']}")
                st.write(f"Efficiency: {model['efficiency']}%")
                st.write(f"Contribution: {model['contribution']}")
                st.write("---")
        
        with col2:
            for model in world_models[2:]:
                st.markdown(f"**{model['site']}**")
                st.write(f"Assets: {model['assets']}")
                st.write(f"Processes: {model['processes']}")
                st.write(f"Efficiency: {model['efficiency']}%")
                st.write(f"Contribution: {model['contribution']}")
                st.write("---")
        
        # Unified metrics
        st.subheader("🌐 Unified Global Intelligence")
        total_assets = sum(model['assets'] for model in world_models)
        total_processes = sum(model['processes'] for model in world_models)
        avg_efficiency = sum(model['efficiency'] for model in world_models) / len(world_models)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", f"{total_assets:,}")
        with col2:
            st.metric("Total Processes", f"{total_processes}")
        with col3:
            st.metric("Avg Efficiency", f"{avg_efficiency:.1f}%")
        with col4:
            st.metric("Sites Connected", len(world_models))
        
        st.markdown("### 🔗 **Unified System Benefits**")
        st.write("""
        - **Cross-Site Learning** - Best practices shared across facilities
        - **Global Optimization** - System-wide efficiency improvements
        - **Predictive Intelligence** - Pattern recognition across industries
        - **Scalable Architecture** - Each site contributes to collective intelligence
        """)
        
        if st.button("⚠️ Continue to Risk Analysis", key="continue_risk"):
            st.session_state.current_step = 4
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 4:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("⚠️ Risk Analysis")
        st.write("Analyzing discovered assets for potential risks and vulnerabilities...")
        
        with st.spinner("Analyzing risks..."):
            time.sleep(2)
            st.session_state.risk_insights = simulate_risk_analysis()
        
        st.warning(f"⚠️ Found {len(st.session_state.risk_insights)} risk insights")
        
        for risk in st.session_state.risk_insights:
            st.markdown(f"• **{risk}**")
        
        # Create risk visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_levels = ['Low', 'Medium', 'High']
        risk_counts = [random.randint(1, 3), random.randint(2, 4), random.randint(1, 2)]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        
        bars = ax.bar(risk_levels, risk_counts, color=colors)
        ax.set_title('Risk Distribution')
        ax.set_ylabel('Number of Assets')
        
        for bar, count in zip(bars, risk_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        risk_chart = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        st.image(f"data:image/png;base64,{risk_chart}", caption="Risk Distribution")
        
        if st.button("🤖 Continue to AI Recommendations", key="continue_ai"):
            st.session_state.current_step = 5
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 5:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.header("🤖 AI Recommendations")
        st.write("Generating AI-powered recommendations for optimization...")
        
        with st.spinner("Generating recommendations..."):
            time.sleep(2)
            st.session_state.recommendations = simulate_recommendations()
        
        st.success(f"✅ Generated {len(st.session_state.recommendations)} recommendations")
        
        for i, rec in enumerate(st.session_state.recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Create efficiency improvement chart
        fig, ax = plt.subplots(figsize=(8, 4))
        improvements = [random.randint(20, 40) for _ in range(5)]
        labels = ['Efficiency', 'Reliability', 'Safety', 'Cost', 'Performance']
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        
        bars = ax.bar(labels, improvements, color=colors)
        ax.set_title('Expected Improvements (%)')
        ax.set_ylabel('Improvement %')
        
        for bar, improvement in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f"{improvement}%", ha='center', va='bottom')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        improvement_chart = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        st.image(f"data:image/png;base64,{improvement_chart}", caption="Expected Improvements")
        
        if st.button("📦 Continue to Export", key="continue_export"):
            st.session_state.current_step = 6
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_step == 6:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.header("📦 World Model Export")
        st.write("Exporting encrypted world model for secure deployment...")
        
        with st.spinner("Exporting data..."):
            time.sleep(2)
            export_data = export_world_model()
        
        st.success("✅ Export completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Export Size", f"{export_data['export_size']:,} bytes")
        with col2:
            st.metric("Assets Exported", export_data['total_assets'])
        with col3:
            st.metric("Sites Included", export_data['sites'])
        
        st.info("🔐 **Encrypted Export Ready**")
        st.write("""
        The world model has been encrypted and exported successfully. 
        This data can now be deployed to your OT environment for real-time monitoring.
        """)
        
        if st.button("🔄 Restart Demo", key="restart_demo"):
            st.session_state.current_step = 0
            st.session_state.discovered_assets = []
            st.session_state.risk_insights = []
            st.session_state.recommendations = []
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 