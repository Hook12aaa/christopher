"""
Rich Logger Utility - Enhanced logging with console and file output

This module provides a sophisticated logging setup using Rich for beautiful
console output and structured file logging for the Sysnpire project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.theme import Theme

# Install rich traceback handler for better error display
install(show_locals=True)

# Custom theme for consistent branding
SYSNPIRE_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "debug": "dim cyan",
    "timestamp": "bright_black",
    "module": "blue",
    "charge": "magenta",
    "field": "bright_magenta",
    "embedding": "green",
    "device": "bright_yellow"
})

class SysnpireLogger:
    """
    Enhanced logging utility for the Sysnpire project.
    
    Features:
    - Rich console output with syntax highlighting
    - File logging with rotation
    - Structured logging for different components
    - Performance monitoring
    - Device and model tracking
    - Simple logging methods (log_info, log_warning, etc.)
    """
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _default_logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.console = Console(theme=SYSNPIRE_THEME)
        
        # Create logs directory
        # Get absolute path to Sysnpire/logs
        module_path = Path(__file__).parent.parent  # Goes to Sysnpire/
        self.log_dir = module_path / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup base configuration
        self._setup_base_logging()
        
        # Create default logger for simple logging
        self._default_logger = self.get_logger('sysnpire')
    
    def _setup_base_logging(self):
        """Setup base logging configuration."""
        # Remove any existing handlers
        logging.getLogger().handlers = []
        
        # Set base level
        logging.getLogger().setLevel(logging.DEBUG)
    
    def get_logger(self, 
                   name: str, 
                   log_file: Optional[str] = None,
                   console_level: int = logging.INFO,
                   file_level: int = logging.DEBUG) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name (str): Logger name (typically module name)
            log_file (str, optional): Custom log file name
            console_level (int): Console logging level
            file_level (int): File logging level
        
        Returns:
            logging.Logger: Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Rich console handler
        console_handler = RichHandler(
            console=self.console,
            show_path=True,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True
        )
        console_handler.setLevel(console_level)
        
        # Custom formatter for console
        console_formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with structured logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"sysnpire_{timestamp}.log"
        
        file_path = self.log_dir / log_file
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Cache logger
        self._loggers[name] = logger
        
        return logger
    
    def log_device_info(self, logger: logging.Logger, device: str, device_name: str = ""):
        """Log device selection information."""
        if "cuda" in device:
            logger.info(f"[device]🚀 Using CUDA GPU: {device_name}[/device]")
        elif "mps" in device:
            logger.info(f"[device]🍎 Using Apple Silicon MPS[/device]")
        else:
            logger.info(f"[device]💻 Using CPU[/device]")
    
    def log_model_info(self, logger: logging.Logger, model_name: str, embedding_dim: int):
        """Log model loading information."""
        logger.info(f"[embedding]🧠 Loading model: {model_name}[/embedding]")
        logger.info(f"[embedding]📐 Embedding dimension: {embedding_dim}[/embedding]")
    
    def log_embedding_stats(self, logger: logging.Logger, text_count: int, cache_size: int):
        """Log embedding generation statistics."""
        logger.info(f"[embedding]📊 Generated embeddings: {text_count}[/embedding]")
        logger.info(f"[embedding]💾 Cache size: {cache_size}[/embedding]")
    
    def log_charge_generation(self, logger: logging.Logger, charge_magnitude: float, phase: float):
        """Log conceptual charge generation."""
        logger.info(f"[charge]⚡ Charge magnitude: {charge_magnitude:.4f}[/charge]")
        logger.info(f"[field]🌊 Phase: {phase:.4f}[/field]")
    
    def log_performance(self, logger: logging.Logger, operation: str, duration: float, items: int = 1):
        """Log performance metrics."""
        rate = items / duration if duration > 0 else 0
        logger.info(f"[success]⏱️  {operation}: {duration:.3f}s ({rate:.1f} items/s)[/success]")
    
    def log_error_context(self, logger: logging.Logger, error: Exception, context: Dict[str, Any]):
        """Log error with rich context information."""
        logger.error(f"[error]❌ Error: {str(error)}[/error]")
        for key, value in context.items():
            logger.error(f"[error]   {key}: {value}[/error]")
    
    def set_module_level(self, module: str, level: int):
        """Set logging level for specific module."""
        if module in self._loggers:
            self._loggers[module].setLevel(level)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "active_loggers": len(self._loggers),
            "log_directory": str(self.log_dir),
            "log_files": [f.name for f in self.log_dir.glob("*.log")]
        }
    
    # Simple logging methods
    def log_info(self, message: str, **kwargs):
        """Log info message with Rich formatting."""
        self._default_logger.info(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with Rich formatting."""
        self._default_logger.warning(message, extra=kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message with Rich formatting."""
        self._default_logger.error(message, extra=kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with Rich formatting."""
        self._default_logger.debug(message, extra=kwargs)
    
    def log_critical(self, message: str, **kwargs):
        """Log critical message with Rich formatting."""
        self._default_logger.critical(message, extra=kwargs)
    
    def log_success(self, message: str, **kwargs):
        """Log success message with Rich formatting (info level with success styling)."""
        self._default_logger.info(f"[success]{message}[/success]", extra=kwargs)

# Global logger instance
_logger_instance = SysnpireLogger()

# Export simple logging methods
log_info = _logger_instance.log_info
log_warning = _logger_instance.log_warning
log_error = _logger_instance.log_error
log_debug = _logger_instance.log_debug
log_critical = _logger_instance.log_critical
log_success = _logger_instance.log_success

def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name (str): Logger name
        **kwargs: Additional arguments for logger configuration
    
    Returns:
        logging.Logger: Configured logger
    """
    return _logger_instance.get_logger(name, **kwargs)

def log_device_info(logger: logging.Logger, device: str, device_name: str = ""):
    """Convenience function for device logging."""
    _logger_instance.log_device_info(logger, device, device_name)

def log_model_info(logger: logging.Logger, model_name: str, embedding_dim: int):
    """Convenience function for model logging."""
    _logger_instance.log_model_info(logger, model_name, embedding_dim)

def log_embedding_stats(logger: logging.Logger, text_count: int, cache_size: int):
    """Convenience function for embedding stats logging."""
    _logger_instance.log_embedding_stats(logger, text_count, cache_size)

def log_performance(logger: logging.Logger, operation: str, duration: float, items: int = 1):
    """Convenience function for performance logging."""
    _logger_instance.log_performance(logger, operation, duration, items)

def log_error_context(logger: logging.Logger, error: Exception, context: Dict[str, Any]):
    """Convenience function for error context logging."""
    _logger_instance.log_error_context(logger, error, context)