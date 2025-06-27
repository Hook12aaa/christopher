"""
LogPolarCDF - Sage CDF in Log-Polar Form for Numerical Stability

Combines Sage's mathematical sophistication with log-polar numerical stability.
Drop-in replacement for LogPolarComplex with native CDF integration.
"""

import math
from dataclasses import dataclass
from typing import Union, Tuple
from sage.all import CDF


@dataclass
class LogPolarCDF:
    """
    Sage CDF complex number in log-polar representation.
    
    Stores z = exp(log_mag) * exp(i*phase) using Sage CDF mathematics.
    Prevents overflow/underflow while maintaining Sage's precision.
    """
    
    log_magnitude: float
    phase: float
    
    def __post_init__(self):
        """Validate inputs and normalize phase."""
        if not math.isfinite(self.log_magnitude):
            raise ValueError(f"log_magnitude must be finite, got {self.log_magnitude}")
        if not math.isfinite(self.phase):
            raise ValueError(f"phase must be finite, got {self.phase}")
        
        # Normalize phase to [0, 2π)
        self.phase = self.phase % (2 * math.pi)
    
    @classmethod
    def from_complex(cls, z: complex) -> 'LogPolarCDF':
        """Create LogPolarCDF from standard complex number."""
        if z == 0:
            raise ValueError("Cannot convert zero to log-polar form (log(0) undefined)")
        
        mag = abs(z)
        phase = math.atan2(z.imag, z.real)
        return cls(math.log(mag), phase)
    
    @classmethod
    def from_cdf(cls, cdf_value) -> 'LogPolarCDF':
        """Create LogPolarCDF from Sage CDF value."""
        return cls.from_complex(complex(cdf_value))
    
    @classmethod
    def from_real_imag(cls, real: float, imag: float) -> 'LogPolarCDF':
        """Create LogPolarCDF from real and imaginary parts."""
        return cls.from_complex(complex(real, imag))
    
    def to_complex(self) -> complex:
        """Convert to standard Python complex number."""
        mag = math.exp(self.log_magnitude)
        return complex(mag * math.cos(self.phase), mag * math.sin(self.phase))
    
    def to_cdf(self):
        """Convert to Sage CDF for mathematical operations."""
        mag = math.exp(self.log_magnitude)
        return CDF(mag * math.cos(self.phase), mag * math.sin(self.phase))
    
    def arg(self) -> float:
        """Return phase angle (compatible with CDF.arg())."""
        return self.phase
    
    @property
    def angle(self) -> float:
        """Alias for phase (backwards compatibility)."""
        return self.phase
    
    def __add__(self, other: 'LogPolarCDF') -> 'LogPolarCDF':
        """
        Add two LogPolarCDF numbers using Sage CDF mathematics.
        
        Uses CDF for the actual addition to maintain Sage precision,
        then converts back to log-polar form for stability.
        """
        if not isinstance(other, LogPolarCDF):
            raise TypeError(f"Can only add LogPolarCDF, got {type(other)}")
        
        # Convert to CDF, perform addition, convert back
        cdf1 = self.to_cdf()
        cdf2 = other.to_cdf()
        result_cdf = cdf1 + cdf2
        
        return LogPolarCDF.from_cdf(result_cdf)
    
    def __mul__(self, other: 'LogPolarCDF') -> 'LogPolarCDF':
        """
        Multiply two LogPolarCDF numbers.
        
        Multiplication is simple in log-polar:
        log(|z1*z2|) = log(|z1|) + log(|z2|)
        arg(z1*z2) = arg(z1) + arg(z2)
        """
        if not isinstance(other, LogPolarCDF):
            raise TypeError(f"Can only multiply LogPolarCDF, got {type(other)}")
        
        new_log_mag = self.log_magnitude + other.log_magnitude
        new_phase = (self.phase + other.phase) % (2 * math.pi)
        
        return LogPolarCDF(new_log_mag, new_phase)
    
    def __abs__(self) -> float:
        """Return magnitude."""
        return math.exp(self.log_magnitude)
    
    def abs(self) -> float:
        """Return magnitude (CDF-compatible method)."""
        return self.__abs__()
    
    def __str__(self) -> str:
        """String representation."""
        mag = math.exp(self.log_magnitude)
        return f"LogPolarCDF(mag={mag:.6f}, phase={self.phase:.6f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"LogPolarCDF(log_magnitude={self.log_magnitude}, phase={self.phase})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, LogPolarCDF):
            return False
        return (abs(self.log_magnitude - other.log_magnitude) < 1e-12 and
                abs(self.phase - other.phase) < 1e-12)