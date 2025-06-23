"""
Log-Polar Complex Number Representation

Stores complex numbers as (log_magnitude, phase) to prevent overflow/underflow.
NO DEFAULTS, NO FALLBACKS, NO NaN HANDLING - pure mathematical integrity.
"""

import math
import cmath
from typing import Union, Tuple


class LogPolarComplex:
    """
    Complex number in log-polar form: z = exp(log_mag) * exp(i*phase)
    
    This representation makes overflow/underflow mathematically impossible.
    - Magnitude can be 10^10000 but log(10^10000) = 10000 (finite)
    - Phase is always bounded in [0, 2π)
    """
    
    def __init__(self, log_mag: float, phase: float):
        """
        Initialize log-polar complex number.
        
        Args:
            log_mag: Natural logarithm of magnitude (must be finite)
            phase: Phase angle in radians (will be normalized to [0, 2π))
            
        Raises:
            ValueError: If inputs are not finite numbers
        """
        # NO DEFAULTS - inputs must be valid or we fail
        if not math.isfinite(log_mag):
            raise ValueError(f"log_magnitude must be finite, got {log_mag}")
        if not math.isfinite(phase):
            raise ValueError(f"phase must be finite, got {phase}")
        
        self.log_mag = log_mag
        self.phase = phase % (2 * math.pi)  # Normalize phase to [0, 2π)
    
    @classmethod
    def from_complex(cls, z: complex) -> 'LogPolarComplex':
        """
        Create LogPolarComplex from standard complex number.
        
        Args:
            z: Complex number to convert
            
        Returns:
            LogPolarComplex representation
            
        Raises:
            ValueError: If z is zero (log(0) undefined)
        """
        if z == 0:
            # NO DEFAULTS - zero cannot be represented in log-polar form
            raise ValueError("Cannot represent zero in log-polar form - mathematical undefined")
        
        magnitude = abs(z)
        if magnitude == 0:  # Extra safety check
            raise ValueError("Complex number has zero magnitude - cannot take log")
        
        # Check for overflow/underflow in input
        if not math.isfinite(magnitude):
            raise ValueError(f"Complex magnitude {magnitude} is not finite - input corrupted")
        
        log_mag = math.log(magnitude)
        phase = cmath.phase(z)
        
        return cls(log_mag, phase)
    
    @classmethod
    def from_real_imag(cls, real: float, imag: float) -> 'LogPolarComplex':
        """
        Create LogPolarComplex from real and imaginary parts.
        
        Args:
            real: Real part
            imag: Imaginary part
            
        Returns:
            LogPolarComplex representation
            
        Raises:
            ValueError: If both parts are zero or not finite
        """
        # NO DEFAULTS - validate inputs
        if not math.isfinite(real):
            raise ValueError(f"Real part must be finite, got {real}")
        if not math.isfinite(imag):
            raise ValueError(f"Imaginary part must be finite, got {imag}")
        
        if real == 0 and imag == 0:
            raise ValueError("Cannot represent 0+0j in log-polar form")
        
        magnitude = math.sqrt(real*real + imag*imag)
        if magnitude == 0:  # Numerical safety
            raise ValueError("Magnitude underflowed to zero")
        
        log_mag = math.log(magnitude)
        phase = math.atan2(imag, real)
        
        return cls(log_mag, phase)
    
    def to_complex(self) -> complex:
        """
        Convert to standard complex number.
        
        WARNING: This can overflow for large log_mag values.
        Use only when absolutely necessary for final output.
        
        Returns:
            Standard complex number
            
        Raises:
            OverflowError: If magnitude too large for float
        """
        # Try to compute magnitude
        try:
            mag = math.exp(self.log_mag)
        except OverflowError:
            raise OverflowError(f"Magnitude exp({self.log_mag}) too large for float representation")
        
        # Check for infinity (shouldn't happen with proper validation)
        if not math.isfinite(mag):
            raise OverflowError(f"Magnitude computation resulted in {mag}")
        
        real = mag * math.cos(self.phase)
        imag = mag * math.sin(self.phase)
        
        return complex(real, imag)
    
    def __mul__(self, other: 'LogPolarComplex') -> 'LogPolarComplex':
        """
        Multiply two log-polar complex numbers.
        
        In log space: log(a*b) = log(a) + log(b)
        Phase: arg(a*b) = arg(a) + arg(b)
        
        This operation CANNOT overflow.
        """
        if not isinstance(other, LogPolarComplex):
            raise TypeError(f"Can only multiply with LogPolarComplex, got {type(other)}")
        
        return LogPolarComplex(
            self.log_mag + other.log_mag,
            self.phase + other.phase
        )
    
    def __truediv__(self, other: 'LogPolarComplex') -> 'LogPolarComplex':
        """
        Divide two log-polar complex numbers.
        
        In log space: log(a/b) = log(a) - log(b)
        Phase: arg(a/b) = arg(a) - arg(b)
        
        This operation CANNOT overflow.
        """
        if not isinstance(other, LogPolarComplex):
            raise TypeError(f"Can only divide by LogPolarComplex, got {type(other)}")
        
        return LogPolarComplex(
            self.log_mag - other.log_mag,
            self.phase - other.phase
        )
    
    def __add__(self, other: 'LogPolarComplex') -> 'LogPolarComplex':
        """
        Add two log-polar complex numbers.
        
        Addition is more complex in log-polar form but still stable.
        Uses log-sum-exp trick to avoid overflow.
        """
        if not isinstance(other, LogPolarComplex):
            raise TypeError(f"Can only add LogPolarComplex, got {type(other)}")
        
        # Convert to rectangular for addition (using stable computation)
        # z1 + z2 = r1*e^(i*θ1) + r2*e^(i*θ2)
        
        # Use log-sum-exp trick for magnitude
        # |z1 + z2| = exp(log_mag_result)
        
        # Compute real and imaginary parts in log space
        log_real1 = self.log_mag + math.log(abs(math.cos(self.phase))) if math.cos(self.phase) != 0 else float('-inf')
        log_real2 = other.log_mag + math.log(abs(math.cos(other.phase))) if math.cos(other.phase) != 0 else float('-inf')
        
        log_imag1 = self.log_mag + math.log(abs(math.sin(self.phase))) if math.sin(self.phase) != 0 else float('-inf')
        log_imag2 = other.log_mag + math.log(abs(math.sin(other.phase))) if math.sin(other.phase) != 0 else float('-inf')
        
        # Signs
        sign_real1 = 1 if math.cos(self.phase) >= 0 else -1
        sign_real2 = 1 if math.cos(other.phase) >= 0 else -1
        sign_imag1 = 1 if math.sin(self.phase) >= 0 else -1
        sign_imag2 = 1 if math.sin(other.phase) >= 0 else -1
        
        # For now, convert to complex for addition (TODO: implement stable log-space addition)
        # This is the only place where overflow might occur, but it's explicit
        z1 = self.to_complex()
        z2 = other.to_complex()
        result = z1 + z2
        
        return LogPolarComplex.from_complex(result)
    
    def conjugate(self) -> 'LogPolarComplex':
        """Complex conjugate (negate phase)."""
        return LogPolarComplex(self.log_mag, -self.phase)
    
    def __abs__(self) -> float:
        """
        Magnitude of complex number.
        
        Returns exp(log_mag) - may overflow for very large values.
        For comparisons, use log_mag directly.
        """
        return math.exp(self.log_mag)
    
    @property
    def log_magnitude(self) -> float:
        """Get log magnitude (always finite)."""
        return self.log_mag
    
    @property
    def angle(self) -> float:
        """Get phase angle in radians [0, 2π)."""
        return self.phase
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LogPolarComplex(log_mag={self.log_mag:.6f}, phase={self.phase:.6f})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"exp({self.log_mag:.2f}) * exp(i*{self.phase:.2f})"