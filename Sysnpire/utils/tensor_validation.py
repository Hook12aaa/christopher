"""
Tensor Validation Utilities - Safe PyTorch Tensor Operations

Provides bulletproof tensor validation and boolean evaluation functions to prevent
"Boolean value of Tensor with more than one value is ambiguous" errors with
ZERO TOLERANCE for unsafe tensor operations.

Key Features:
- Safe tensor boolean evaluation
- Explicit size and finite value validation  
- Comprehensive error messaging
- NO silent failures or implicit conversions
"""

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TensorValidationError(Exception):
    """Raised when tensor validation fails with detailed error information."""
    pass


def safe_tensor_boolean(tensor: torch.Tensor, context: str = "") -> bool:
    """
    Safely convert tensor to boolean with explicit validation.
    
    Prevents "Boolean value of Tensor with more than one value is ambiguous" errors
    by providing explicit, unambiguous tensor boolean evaluation.
    
    Args:
        tensor: PyTorch tensor to evaluate
        context: Context string for error reporting
        
    Returns:
        bool: True if tensor is non-empty and has valid data, False otherwise
        
    Raises:
        TensorValidationError: If tensor evaluation fails
    """
    if tensor is None:
        return False
        
    try:
        # CRITICAL: Explicit size check BEFORE any boolean operations
        tensor_size = tensor.numel()
        
        if tensor_size == 0:
            logger.debug(f"Tensor is empty in context: {context}")
            return False
        elif tensor_size == 1:
            # Safe to extract scalar value
            scalar_value = tensor.cpu().item()
            return bool(scalar_value)
        else:
            # Multi-element tensor - check if ANY values are non-zero
            # This is explicit and unambiguous
            cpu_tensor = tensor.cpu()
            has_nonzero = torch.any(cpu_tensor != 0).item()
            logger.debug(f"Multi-element tensor ({tensor_size} elements) has non-zero values: {has_nonzero} in context: {context}")
            return bool(has_nonzero)
            
    except Exception as e:
        error_msg = f"Tensor boolean evaluation failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def validate_tensor_finite(tensor: torch.Tensor, context: str = "") -> bool:
    """
    Validate that tensor contains only finite values with explicit size checking.
    
    Args:
        tensor: PyTorch tensor to validate
        context: Context string for error reporting
        
    Returns:
        bool: True if tensor has finite values, False otherwise
        
    Raises:
        TensorValidationError: If validation cannot be performed
    """
    if tensor is None:
        raise TensorValidationError(f"Cannot validate None tensor in context: {context}")
        
    try:
        # CRITICAL: Check size BEFORE any operations
        tensor_size = tensor.numel()
        
        if tensor_size == 0:
            logger.debug(f"Empty tensor considered finite in context: {context}")
            return True
        
        # Move to CPU for safe validation (avoids MPS float64 issues)
        cpu_tensor = tensor.cpu()
        
        # EXPLICIT finite check with unambiguous boolean conversion
        finite_mask = torch.isfinite(cpu_tensor)
        all_finite = torch.all(finite_mask).item()  # Explicit .item() conversion
        
        if not all_finite:
            # Count non-finite values for detailed error reporting
            num_non_finite = torch.sum(~finite_mask).item()
            logger.error(f"💥 Tensor has {num_non_finite}/{tensor_size} non-finite values in context: {context}")
        
        return bool(all_finite)
        
    except Exception as e:
        error_msg = f"Tensor finite validation failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def safe_tensor_size_check(tensor: torch.Tensor, context: str = "") -> int:
    """
    Safely get tensor size with comprehensive error handling.
    
    Args:
        tensor: PyTorch tensor to check
        context: Context string for error reporting
        
    Returns:
        int: Number of elements in tensor
        
    Raises:
        TensorValidationError: If size check fails
    """
    if tensor is None:
        raise TensorValidationError(f"Cannot get size of None tensor in context: {context}")
        
    try:
        tensor_size = tensor.numel()
        logger.debug(f"Tensor size: {tensor_size} elements in context: {context}")
        return tensor_size
        
    except Exception as e:
        error_msg = f"Tensor size check failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def extract_tensor_scalar(tensor: torch.Tensor, context: str = "") -> Union[float, complex]:
    """
    Safely extract scalar value from single-element tensor.
    
    Args:
        tensor: PyTorch tensor that should contain exactly one element
        context: Context string for error reporting
        
    Returns:
        Scalar value (float or complex)
        
    Raises:
        TensorValidationError: If tensor is not single-element or extraction fails
    """
    if tensor is None:
        raise TensorValidationError(f"Cannot extract scalar from None tensor in context: {context}")
        
    try:
        tensor_size = tensor.numel()
        
        if tensor_size == 0:
            raise TensorValidationError(f"Cannot extract scalar from empty tensor in context: {context}")
        elif tensor_size > 1:
            raise TensorValidationError(f"Cannot extract scalar from multi-element tensor ({tensor_size} elements) in context: {context}")
        
        # Safe scalar extraction
        scalar_value = tensor.cpu().item()
        logger.debug(f"Extracted scalar value: {scalar_value} in context: {context}")
        return scalar_value
        
    except Exception as e:
        error_msg = f"Tensor scalar extraction failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def validate_tensor_operations(tensor: torch.Tensor, operation_name: str, context: str = "") -> None:
    """
    Pre-validate tensor before performing operations.
    
    Args:
        tensor: PyTorch tensor to validate
        operation_name: Name of operation to be performed
        context: Context string for error reporting
        
    Raises:
        TensorValidationError: If tensor is not suitable for operations
    """
    if tensor is None:
        raise TensorValidationError(f"Cannot perform {operation_name} on None tensor in context: {context}")
        
    try:
        # Check basic tensor properties
        tensor_size = tensor.numel()
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype
        tensor_device = tensor.device
        
        logger.debug(f"Validating tensor for {operation_name} in context: {context}")
        logger.debug(f"  Size: {tensor_size}, Shape: {tensor_shape}, Dtype: {tensor_dtype}, Device: {tensor_device}")
        
        # Validate tensor has data
        if tensor_size == 0:
            raise TensorValidationError(f"Cannot perform {operation_name} on empty tensor in context: {context}")
        
        # Validate finite values
        if not validate_tensor_finite(tensor, f"{context}.{operation_name}"):
            raise TensorValidationError(f"Cannot perform {operation_name} on tensor with non-finite values in context: {context}")
        
        logger.debug(f"✅ Tensor validation passed for {operation_name} in context: {context}")
        
    except TensorValidationError:
        # Re-raise tensor validation errors as-is
        raise
    except Exception as e:
        error_msg = f"Tensor operation validation failed for {operation_name} in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def tensor_to_python_safe(tensor: torch.Tensor, context: str = "") -> Union[float, complex, np.ndarray]:
    """
    Safely convert tensor to Python/NumPy representation.
    
    Args:
        tensor: PyTorch tensor to convert
        context: Context string for error reporting
        
    Returns:
        Python scalar (for single-element tensors) or NumPy array
        
    Raises:
        TensorValidationError: If conversion fails
    """
    if tensor is None:
        raise TensorValidationError(f"Cannot convert None tensor in context: {context}")
        
    try:
        tensor_size = tensor.numel()
        
        if tensor_size == 0:
            logger.debug(f"Converting empty tensor to empty numpy array in context: {context}")
            return tensor.cpu().detach().numpy()
        elif tensor_size == 1:
            # Extract scalar
            scalar_value = extract_tensor_scalar(tensor, f"{context}.scalar_conversion")
            logger.debug(f"Converted single-element tensor to scalar: {scalar_value} in context: {context}")
            return scalar_value
        else:
            # Convert to numpy array
            numpy_array = tensor.cpu().detach().numpy()
            logger.debug(f"Converted multi-element tensor to numpy array: shape={numpy_array.shape} in context: {context}")
            return numpy_array
            
    except TensorValidationError:
        # Re-raise tensor validation errors as-is
        raise
    except Exception as e:
        error_msg = f"Tensor to Python conversion failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def safe_tensor_comparison(tensor: torch.Tensor, comparison_value: Union[int, float], context: str = "") -> bool:
    """
    Safely compare tensor size to a value without ambiguous boolean evaluation.
    
    Args:
        tensor: PyTorch tensor to check
        comparison_value: Value to compare tensor size against
        context: Context string for error reporting
        
    Returns:
        bool: True if tensor.numel() == comparison_value
        
    Raises:
        TensorValidationError: If comparison fails
    """
    if tensor is None:
        return False
        
    try:
        tensor_size = safe_tensor_size_check(tensor, f"{context}.size_comparison")
        result = tensor_size == comparison_value
        logger.debug(f"Tensor size comparison: {tensor_size} == {comparison_value} = {result} in context: {context}")
        return result
        
    except Exception as e:
        error_msg = f"Tensor size comparison failed in context '{context}': {e}"
        logger.error(f"❌ {error_msg}")
        raise TensorValidationError(error_msg) from e


def validate_non_empty_tensor(tensor: torch.Tensor, context: str = "") -> bool:
    """
    Validate that tensor is not None and has elements.
    
    Args:
        tensor: PyTorch tensor to validate
        context: Context string for error reporting
        
    Returns:
        bool: True if tensor is not None and has elements
    """
    if tensor is None:
        logger.debug(f"Tensor is None in context: {context}")
        return False
        
    try:
        tensor_size = safe_tensor_size_check(tensor, f"{context}.non_empty_check")
        is_non_empty = tensor_size > 0
        logger.debug(f"Tensor non-empty check: size={tensor_size}, non_empty={is_non_empty} in context: {context}")
        return is_non_empty
        
    except Exception as e:
        logger.warning(f"⚠️ Tensor non-empty validation failed in context '{context}': {e}")
        return False


if __name__ == "__main__":
    # Test safe tensor validation utilities
    print("Testing Tensor Validation Utilities...")
    
    # Test with various tensor types
    test_tensors = [
        torch.tensor([1.0]),                    # Single element
        torch.tensor([1.0, 2.0, 3.0]),        # Multi-element
        torch.tensor([]),                       # Empty
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # 2D
    ]
    
    for i, tensor in enumerate(test_tensors):
        try:
            print(f"\nTensor {i}: shape={tensor.shape}")
            print(f"  Safe boolean: {safe_tensor_boolean(tensor, f'test_{i}')}")
            print(f"  Is finite: {validate_tensor_finite(tensor, f'test_{i}')}")
            print(f"  Size: {safe_tensor_size_check(tensor, f'test_{i}')}")
            print(f"  Non-empty: {validate_non_empty_tensor(tensor, f'test_{i}')}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✅ Tensor validation utilities ready!")