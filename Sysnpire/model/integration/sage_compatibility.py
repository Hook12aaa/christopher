"""
SAGE Compatibility Layer for Integration Module

This module provides compatibility functions for handling SAGE mathematical objects
within the integration layer. The main codebase uses SAGE for mathematical precision,
and this layer ensures smooth conversion between SAGE objects and PyTorch tensors.

CRITICAL: This must match the patterns used in the main codebase (conceptual_charge_agent.py)
to ensure consistent behavior when processing universe data.
"""

import numpy as np
import torch


def safe_torch_tensor(data, **kwargs):
    """
    Create torch tensor safely from data that may contain SAGE objects.
    
    Automatically converts SAGE ComplexDoubleElement, Integer, RealDoubleElement
    to Python primitives before passing to torch.tensor().
    
    This is a direct copy from the main codebase to ensure consistency.
    
    Args:
        data: Input data (may contain SAGE objects)
        **kwargs: Arguments to pass to torch.tensor()
        
    Returns:
        torch.Tensor with SAGE objects converted to Python primitives
    """
    def sage_to_python(value):
        """Convert single SAGE object to Python primitive."""
        # Import SAGE types for type checking
        try:
            from sage.rings.complex_double import ComplexDoubleElement
            from sage.rings.integer import Integer as SageInteger
            from sage.rings.real_double import RealDoubleElement
        except ImportError:
            return value
            
        # Convert SAGE ComplexDoubleElement to Python complex
        if isinstance(value, ComplexDoubleElement):
            return complex(float(value.real()), float(value.imag()))
            
        # Convert SAGE Integer to Python int  
        if isinstance(value, SageInteger):
            return int(value)
            
        # Convert SAGE RealDoubleElement to Python float
        if isinstance(value, RealDoubleElement):
            return float(value)
            
        # Check for any other SAGE types
        if hasattr(value, '__class__') and 'sage' in str(type(value)):
            if hasattr(value, 'real') and hasattr(value, 'imag'):
                return complex(float(value.real()), float(value.imag()))
            elif hasattr(value, '__float__'):
                return float(value)
            elif hasattr(value, '__int__'):
                return int(value)
                
        return value
    
    # Handle different data types
    if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
        # Handle arrays/lists - recursively convert elements
        if isinstance(data, np.ndarray):
            # For numpy arrays, apply conversion element-wise
            converted_data = np.array([sage_to_python(item) for item in data.flat]).reshape(data.shape)
        else:
            # For lists/tuples, convert elements
            converted_data = [sage_to_python(item) for item in data]
    else:
        # Handle single values
        converted_data = sage_to_python(data)
    
    return torch.tensor(converted_data, **kwargs)


def convert_sage_dict(data_dict):
    """
    Recursively convert a dictionary that may contain SAGE objects.
    
    Useful for converting entire agent states or universe configurations
    that contain nested SAGE mathematical objects.
    
    Args:
        data_dict: Dictionary potentially containing SAGE objects
        
    Returns:
        Dictionary with all SAGE objects converted to Python primitives
    """
    if not isinstance(data_dict, dict):
        return data_dict
        
    converted = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            converted[key] = convert_sage_dict(value)
        elif isinstance(value, (list, tuple)):
            converted[key] = type(value)(safe_torch_tensor(item).item() if torch.is_tensor(safe_torch_tensor(item)) else item for item in value)
        else:
            # Try to convert to tensor and extract value
            try:
                tensor_val = safe_torch_tensor(value)
                if torch.is_tensor(tensor_val) and tensor_val.numel() == 1:
                    converted[key] = tensor_val.item()
                else:
                    converted[key] = value
            except:
                converted[key] = value
                
    return converted


def ensure_python_types(value):
    """
    Ensure a value is a Python primitive type, not a SAGE object.
    
    This is a lightweight check for single values that need to be
    Python primitives for mathematical operations.
    
    Args:
        value: Value to check and convert if needed
        
    Returns:
        Python primitive (complex, float, int) or original value
    """
    try:
        from sage.rings.complex_double import ComplexDoubleElement
        from sage.rings.integer import Integer as SageInteger
        from sage.rings.real_double import RealDoubleElement
        
        if isinstance(value, ComplexDoubleElement):
            return complex(float(value.real()), float(value.imag()))
        elif isinstance(value, SageInteger):
            return int(value)
        elif isinstance(value, RealDoubleElement):
            return float(value)
    except ImportError:
        pass
        
    return value