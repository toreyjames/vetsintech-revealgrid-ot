#!/usr/bin/env python3
"""
Test script for RevealGrid MVP
Verifies all components work correctly together
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import networkx as nx
        print("✅ networkx imported successfully")
    except ImportError as e:
        print(f"❌ networkx import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import sympy as sp
        print("✅ sympy imported successfully")
    except ImportError as e:
        print(f"❌ sympy import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import revealgrid_mvp as rg
        print("✅ revealgrid_mvp imported successfully")
    except ImportError as e:
        print(f"❌ revealgrid_mvp import failed: {e}")
        return False
    
    return True

def test_core_functions():
    """Test core RevealGrid functions"""
    print("\nTesting core functions...")
    
    try:
        import revealgrid_mvp as rg
        
        # Test site mapping
        site_map_viz = rg.generate_site_map()
        print(f"✅ Site map generation: {len(site_map_viz)} base64 characters")
        
        # Test global metrics
        global_metrics = rg.get_global_metrics()
        print(f"✅ Global metrics: {global_metrics['total_sites']} sites, {global_metrics['avg_efficiency']:.1f}% avg efficiency")
        
        # Test asset discovery
        discovered_assets, asset_categories = rg.discover_assets()
        print(f"✅ Asset discovery: {len(discovered_assets)} assets found")
        
        # Test graph creation
        graph = rg.create_sample_graph()
        print(f"✅ Graph creation: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Test recommendations
        recommendations, risks = rg.recommend_automation(graph, use_rl_sim=True)
        print(f"✅ Automation recommendations: {len(recommendations)} recommendations, {len(risks)} risks")
        
        # Test world model
        world_sim = rg.build_world_model(graph)
        print(f"✅ World model simulation: {len(world_sim['optimization_steps'])} steps")
        
        # Test robot actions
        robot_actions, _ = rg.robot_use_model(graph)
        print(f"✅ Robot actions: {len(robot_actions)} actions planned")
        
        # Test visualization
        viz = rg.visualize_graph(graph, risks)
        print(f"✅ Graph visualization: {len(viz)} base64 characters")
        
        # Test world model export
        world_model = rg.export_world_model(graph, world_sim, robot_actions)
        print(f"✅ World model export: {len(world_model['graph']['nodes'])} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functions test failed: {e}")
        traceback.print_exc()
        return False

def test_site_mapping():
    """Test site mapping functionality"""
    print("\nTesting site mapping...")
    
    try:
        import revealgrid_mvp as rg
        
        # Test site data
        print(f"✅ Site data: {len(rg.sites)} sites loaded")
        for site_name, site_data in rg.sites.items():
            print(f"   - {site_name}: {site_data['efficiency']}% efficiency, {site_data['risk_score']}% risk")
        
        # Test site map generation
        site_map = rg.generate_site_map()
        print(f"✅ Site map generation: {len(site_map)} base64 characters")
        
        # Test drill-down functionality
        test_site = list(rg.sites.keys())[0]
        print(f"✅ Drill-down test for {test_site}")
        rg.generate_site_map(drill_down_site=test_site)
        
        # Test global metrics
        metrics = rg.get_global_metrics()
        required_keys = ['total_sites', 'avg_efficiency', 'avg_downtime', 'avg_risk_score', 'high_risk_sites', 'low_efficiency_sites']
        for key in required_keys:
            if key not in metrics:
                print(f"❌ Missing key in global metrics: {key}")
                return False
        
        print(f"✅ Global metrics: {metrics['total_sites']} sites, {metrics['avg_efficiency']:.1f}% avg efficiency")
        
        return True
        
    except Exception as e:
        print(f"❌ Site mapping test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test Streamlit-specific components"""
    print("\nTesting Streamlit components...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import plotly.graph_objects as go
        print("✅ plotly.graph_objects imported successfully")
        
        import plotly.express as px
        print("✅ plotly.express imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Streamlit component import failed: {e}")
        return False

def test_demo_run():
    """Test the full demo run with site mapping"""
    print("\nTesting full demo run with site mapping...")
    
    try:
        import revealgrid_mvp as rg
        results = rg.run_demo()
        
        # Verify results structure
        required_keys = ['sites', 'global_metrics', 'site_map', 'graph', 'recommendations', 'risks', 'world_sim', 'robot_actions', 'world_model', 'visualization']
        for key in required_keys:
            if key not in results:
                print(f"❌ Missing key in results: {key}")
                return False
        
        print("✅ Full demo run completed successfully")
        print(f"   - Sites: {len(results['sites'])}")
        print(f"   - Graph: {len(results['graph'].nodes)} nodes")
        print(f"   - Recommendations: {len(results['recommendations'])}")
        print(f"   - Risks: {len(results['risks'])}")
        print(f"   - Robot actions: {len(results['robot_actions'])}")
        print(f"   - Global metrics: {results['global_metrics']['total_sites']} sites")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo run failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 RevealGrid MVP Test Suite (with Site Mapping)")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    tests = [
        ("Import Tests", test_imports),
        ("Core Function Tests", test_core_functions),
        ("Site Mapping Tests", test_site_mapping),
        ("Streamlit Component Tests", test_streamlit_components),
        ("Demo Run Test", test_demo_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RevealGrid MVP with Site Mapping is ready to use.")
        print("\nTo run the Streamlit app:")
        print("  streamlit run app.py")
        print("\nTo run the command-line demo:")
        print("  python revealgrid_mvp.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main() 