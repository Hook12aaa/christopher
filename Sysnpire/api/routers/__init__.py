"""
Router modules for the Constructivist Field Theory API.
"""
# Import all blueprint modules
from . import embedding, charges, fields, resonance, viz

# Make the blueprints available for import
__all__ = ["embedding", "charges", "fields", "resonance", "viz"]