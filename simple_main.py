#!/usr/bin/env python3
"""
Simplified Real-time EEG Attention State Analysis System Main Program
Uses matplotlib instead of pyqtgraph to reduce dependencies
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_main_window import main

if __name__ == "__main__":
    main()
