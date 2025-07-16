#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point for RevealGrid OT
This file serves as the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main() 