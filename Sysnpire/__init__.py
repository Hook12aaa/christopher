"""
Sysnpire - Field Theory of Social Constructs Implementation

This package implements the complete Field Theory of Social Constructs
as defined in the research paper, providing tools for conceptual charge
generation, field dynamics, and semantic universe construction.
"""

__version__ = "0.1.0"
__author__ = "Sysnpire Team"

# Core exports
from . import model
from . import database  
from . import api
from . import dashboard
from . import utils

__all__ = [
    'model',
    'database', 
    'api',
    'dashboard',
    'utils'
]