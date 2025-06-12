"""Structured Logging Utilities for MMFakeBench.

This module provides comprehensive logging setup and utilities for the
misinformation detection benchmark system.
"""

import logging
import logging.config
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted JSON string
        """
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class BenchmarkLogger:
    """Main logger class for the benchmark system."""
    
    def __init__(self, 
                 name: str = "mmfakebench",
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 structured_format: bool = False):
        """Initialize the benchmark logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            structured_format: Whether to use structured JSON format
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.structured_format = structured_format
        
        # Create log directory
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Setup formatters
        if self.structured_format:
            formatter = StructuredFormatter()
            console_formatter = StructuredFormatter(include_extra=False)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file:
            # Main log file
            file_handler = logging.FileHandler(
                self.log_dir / f"{self.name}.log",
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Error log file
            error_handler = logging.FileHandler(
                self.log_dir / f"{self.name}_errors.log",
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        return logger
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance for a specific module.
        
        Args:
            module_name: Name of the module requesting the logger
            
        Returns:
            Logger instance
        """
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def log_benchmark_start(self, config: Dict[str, Any]) -> None:
        """Log benchmark execution start.
        
        Args:
            config: Benchmark configuration
        """
        self.logger.info(
            "Benchmark execution started",
            extra={
                'event_type': 'benchmark_start',
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_benchmark_end(self, 
                         duration: float, 
                         total_samples: int,
                         success_count: int,
                         error_count: int) -> None:
        """Log benchmark execution end.
        
        Args:
            duration: Total execution time in seconds
            total_samples: Total number of samples processed
            success_count: Number of successful predictions
            error_count: Number of errors encountered
        """
        self.logger.info(
            "Benchmark execution completed",
            extra={
                'event_type': 'benchmark_end',
                'duration_seconds': duration,
                'total_samples': total_samples,
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_count / max(1, total_samples),
                'samples_per_second': total_samples / max(0.001, duration),
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_module_execution(self, 
                           module_name: str,
                           duration: float,
                           input_data: Dict[str, Any],
                           output_data: Dict[str, Any],
                           success: bool = True,
                           error: Optional[str] = None) -> None:
        """Log module execution details.
        
        Args:
            module_name: Name of the executed module
            duration: Execution time in seconds
            input_data: Input data (sanitized)
            output_data: Output data (sanitized)
            success: Whether execution was successful
            error: Error message if execution failed
        """
        log_data = {
            'event_type': 'module_execution',
            'module_name': module_name,
            'duration_seconds': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add sanitized input/output data
        log_data['input_summary'] = self._sanitize_data(input_data)
        if success:
            log_data['output_summary'] = self._sanitize_data(output_data)
        else:
            log_data['error'] = error
        
        level = logging.INFO if success else logging.ERROR
        message = f"Module {module_name} executed {'successfully' if success else 'with error'}"
        
        self.logger.log(level, message, extra=log_data)
    
    def log_api_call(self, 
                    provider: str,
                    model: str,
                    tokens_used: int,
                    cost: float,
                    duration: float,
                    success: bool = True,
                    error: Optional[str] = None) -> None:
        """Log API call details.
        
        Args:
            provider: API provider name
            model: Model name
            tokens_used: Number of tokens used
            cost: Cost in USD
            duration: Call duration in seconds
            success: Whether call was successful
            error: Error message if call failed
        """
        log_data = {
            'event_type': 'api_call',
            'provider': provider,
            'model': model,
            'tokens_used': tokens_used,
            'cost_usd': cost,
            'duration_seconds': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if not success and error:
            log_data['error'] = error
        
        level = logging.INFO if success else logging.WARNING
        message = f"API call to {provider}/{model} {'completed' if success else 'failed'}"
        
        self.logger.log(level, message, extra=log_data)
    
    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.logger.info(
            "Performance metrics calculated",
            extra={
                'event_type': 'performance_metrics',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _sanitize_data(self, data: Dict[str, Any], max_length: int = 200) -> Dict[str, Any]:
        """Sanitize data for logging by truncating long values.
        
        Args:
            data: Data to sanitize
            max_length: Maximum length for string values
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                if len(value) > max_length:
                    sanitized[key] = value[:max_length] + "..."
                else:
                    sanitized[key] = value
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                sanitized[key] = str(type(value).__name__)
        
        return sanitized


@contextmanager
def log_execution_time(logger: logging.Logger, operation_name: str):
    """Context manager to log execution time of operations.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation being timed
    """
    start_time = time.time()
    logger.debug(f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"Completed {operation_name}",
            extra={
                'operation': operation_name,
                'duration_seconds': duration,
                'event_type': 'operation_completed'
            }
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation_name}: {str(e)}",
            extra={
                'operation': operation_name,
                'duration_seconds': duration,
                'error': str(e),
                'event_type': 'operation_failed'
            },
            exc_info=True
        )
        raise


def setup_logging(config: Dict[str, Any]) -> BenchmarkLogger:
    """Setup logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured BenchmarkLogger instance
    """
    return BenchmarkLogger(
        name=config.get('name', 'mmfakebench'),
        log_level=config.get('level', 'INFO'),
        log_dir=config.get('log_dir', 'logs'),
        enable_console=config.get('enable_console', True),
        enable_file=config.get('enable_file', True),
        structured_format=config.get('structured_format', False)
    )


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"mmfakebench.{module_name}")


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class.
        
        Returns:
            Logger instance
        """
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            module_name = self.__class__.__module__.split('.')[-1]
            self._logger = get_module_logger(f"{module_name}.{class_name}")
        return self._logger
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with extra data.
        
        Args:
            message: Log message
            **kwargs: Extra data to include
        """
        self.logger.info(message, extra=kwargs)
    
    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with extra data.
        
        Args:
            message: Log message
            error: Exception object if available
            **kwargs: Extra data to include
        """
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        
        self.logger.error(message, extra=kwargs, exc_info=error is not None)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with extra data.
        
        Args:
            message: Log message
            **kwargs: Extra data to include
        """
        self.logger.warning(message, extra=kwargs)