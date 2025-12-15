"""
Urban Issues Detector - YOLO-based object detection for urban problems
"""

__version__ = "1.0.0"
__author__ = "Urban Issues Team"

from . import data_loader
from . import train
from . import eval

__all__ = ['data_loader', 'train', 'eval']
