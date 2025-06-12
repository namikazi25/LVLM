"""Input/Output utilities for benchmark results and logging.

This module provides utilities for saving results, managing logs,
and handling file I/O operations.
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class CircularReferenceEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles circular references."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = set()
    
    def encode(self, obj):
        self.seen = set()
        return super().encode(obj)
    
    def default(self, obj):
        obj_id = id(obj)
        if obj_id in self.seen:
            return "<Circular Reference>"
        
        if isinstance(obj, (dict, list, tuple, set)):
            self.seen.add(obj_id)
            try:
                if isinstance(obj, dict):
                    result = {k: self.default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    result = [self.default(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in obj]
                elif isinstance(obj, set):
                    result = list(obj)
                else:
                    result = str(obj)
                self.seen.remove(obj_id)
                return result
            except:
                self.seen.discard(obj_id)
                return str(obj)
        
        return str(obj)


class CSVExporter:
    """Specialized CSV exporter for benchmark results with comprehensive metrics."""
    
    def __init__(self, output_dir: Union[str, Path] = "results"):
        """Initialize the CSV exporter.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_results(self, results: List[Dict[str, Any]], 
                      filename: Optional[str] = None,
                      include_metadata: bool = True) -> Path:
        """Export benchmark results to CSV with all metrics.
        
        Args:
            results: List of result dictionaries
            filename: Output filename (auto-generated if None)
            include_metadata: Whether to include metadata columns
            
        Returns:
            Path to the exported CSV file
        """
        if not results:
            raise ValueError("Cannot export empty results")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_results_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Prepare data for CSV export
        csv_data = []
        
        for i, result in enumerate(results):
            row = {
                'run_id': i + 1,
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
                'sample_id': result.get('sample_id', f"sample_{i+1}"),
                'prediction': result.get('prediction', ''),
                'ground_truth': result.get('ground_truth', ''),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', 0.0),
                'api_cost': result.get('api_cost', 0.0)
            }
            
            # Add metrics
            metrics = result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                row[f'metric_{metric_name}'] = metric_value
            
            # Add model information
            model_info = result.get('model_info', {})
            row['model_name'] = model_info.get('name', '')
            row['model_version'] = model_info.get('version', '')
            row['model_temperature'] = model_info.get('temperature', 0.0)
            
            # Add dataset information
            dataset_info = result.get('dataset_info', {})
            row['dataset_name'] = dataset_info.get('name', '')
            row['dataset_split'] = dataset_info.get('split', '')
            
            # Add pipeline information
            pipeline_info = result.get('pipeline_info', {})
            row['pipeline_name'] = pipeline_info.get('name', '')
            row['pipeline_version'] = pipeline_info.get('version', '')
            
            if include_metadata:
                # Add configuration details
                config = result.get('config', {})
                for key, value in self._flatten_dict(config, 'config').items():
                    row[key] = value
                
                # Add error information
                errors = result.get('errors', {})
                row['error_count'] = errors.get('count', 0)
                row['error_types'] = str(errors.get('types', []))
            
            csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            fieldnames = list(csv_data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logging.info(f"Exported {len(csv_data)} results to {filepath}")
        return filepath
    
    def export_summary_stats(self, stats: Dict[str, Any], 
                           filename: Optional[str] = None) -> Path:
        """Export summary statistics to CSV.
        
        Args:
            stats: Statistics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the exported CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"summary_stats_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten statistics for CSV
        flattened_stats = self._flatten_dict(stats)
        
        # Convert to list of dictionaries for CSV writer
        csv_data = [{'metric': key, 'value': value} for key, value in flattened_stats.items()]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        logging.info(f"Exported summary statistics to {filepath}")
        return filepath
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_', seen: Optional[set] = None) -> Dict[str, Any]:
        """Flatten a nested dictionary for CSV export.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
            seen: Set to track already-seen dicts to avoid loops
            
        Returns:
            Flattened dictionary
        """
        if seen is None:  # track already-seen dicts to avoid loops
            seen = set()
        if id(d) in seen:
            return {}  # skip circular ref
        seen.add(id(d))
        
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep, seen=seen).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)


class ResultsWriter:
    """Enhanced utility class for writing benchmark results to various formats."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize the results writer.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_exporter = CSVExporter(output_dir)
    
    def save_json(self, data: Union[Dict, List], filename: str) -> Path:
        """Save data as JSON file.
        
        Args:
            data: Data to save
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=CircularReferenceEncoder)
            
            logging.info(f"Saved JSON results to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Failed to save JSON to {filepath}: {e}")
            raise
    
    def save_csv(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """Save data as CSV file.
        
        Args:
            data: List of dictionaries to save
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        if not data:
            raise ValueError("Cannot save empty data to CSV")
        
        filepath = self.output_dir / filename
        
        try:
            # Flatten nested dictionaries for CSV
            flattened_data = [self._flatten_dict(item) for item in data]
            
            # Get all unique keys
            all_keys = set()
            for item in flattened_data:
                all_keys.update(item.keys())
            
            fieldnames = sorted(all_keys)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            logging.info(f"Saved CSV results to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Failed to save CSV to {filepath}: {e}")
            raise
    
    def save_txt(self, content: str, filename: str) -> Path:
        """Save content as text file.
        
        Args:
            content: Text content to save
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info(f"Saved text to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Failed to save text to {filepath}: {e}")
            raise
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_', seen: Optional[set] = None) -> Dict[str, Any]:
        """Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
            seen: Set to track already-seen dicts to avoid loops
            
        Returns:
            Flattened dictionary
        """
        if seen is None:  # track already-seen dicts to avoid loops
            seen = set()
        if id(d) in seen:
            return {}  # skip circular ref
        seen.add(id(d))
        
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep, seen=seen).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def create_summary_report(self, results: List[Dict[str, Any]], 
                            stats: Dict[str, Any]) -> str:
        """Create a summary report from results and statistics.
        
        Args:
            results: List of result dictionaries
            stats: Statistics dictionary
            
        Returns:
            Summary report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("BENCHMARK SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Statistics section
        report_lines.append("STATISTICS:")
        report_lines.append("-" * 20)
        for key, value in stats.items():
            if isinstance(value, float):
                report_lines.append(f"{key}: {value:.4f}")
            else:
                report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Results summary
        if results:
            report_lines.append("RESULTS SUMMARY:")
            report_lines.append("-" * 20)
            report_lines.append(f"Total items processed: {len(results)}")
            
            # Count different result types if available
            if 'prediction' in results[0]:
                predictions = [r.get('prediction') for r in results]
                unique_predictions = set(predictions)
                report_lines.append(f"Unique predictions: {len(unique_predictions)}")
                
                for pred in unique_predictions:
                    count = predictions.count(pred)
                    percentage = (count / len(predictions)) * 100
                    report_lines.append(f"  {pred}: {count} ({percentage:.1f}%)")
            
            report_lines.append("")
        
        # Performance metrics
        if 'duration' in stats and 'processed_items' in stats:
            duration = stats['duration']
            processed = stats['processed_items']
            
            report_lines.append("PERFORMANCE:")
            report_lines.append("-" * 20)
            report_lines.append(f"Total duration: {duration:.2f} seconds")
            report_lines.append(f"Items per second: {processed / duration:.2f}")
            
            if 'total_cost' in stats:
                report_lines.append(f"Total cost: ${stats['total_cost']:.4f}")
                report_lines.append(f"Cost per item: ${stats['total_cost'] / processed:.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


class Logger:
    """Enhanced logging utility for benchmark operations with multiple verbosity levels."""
    
    def __init__(self, log_file: Optional[Union[str, Path]] = None, 
                 level: int = logging.INFO,
                 console_level: Optional[int] = None,
                 enable_progress: bool = True):
        """Initialize the logger with enhanced capabilities.
        
        Args:
            log_file: Path to log file (optional)
            level: File logging level
            console_level: Console logging level (defaults to file level)
            enable_progress: Whether to enable progress indicators
        """
        self.log_file = Path(log_file) if log_file else None
        self.level = level
        self.console_level = console_level or level
        self.enable_progress = enable_progress
        
        # Create logger
        self.logger = logging.getLogger('mmfakebench')
        self.logger.setLevel(min(level, self.console_level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters for different verbosity levels
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        self.standard_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        self.simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Console handler with appropriate formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        
        if self.console_level <= logging.DEBUG:
            console_handler.setFormatter(self.detailed_formatter)
        elif self.console_level <= logging.INFO:
            console_handler.setFormatter(self.standard_formatter)
        else:
            console_handler.setFormatter(self.simple_formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler (if specified) with detailed formatting
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.detailed_formatter)
            self.logger.addHandler(file_handler)
        
        # Progress tracking
        self.current_operation = None
        self.operation_start_time = None
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def start_operation(self, operation_name: str) -> None:
        """Start tracking a long-running operation.
        
        Args:
            operation_name: Name of the operation being started
        """
        import time
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        
        if self.enable_progress:
            self.info(f"Starting {operation_name}...")
    
    def end_operation(self, success: bool = True, details: str = "") -> None:
        """End tracking of the current operation.
        
        Args:
            success: Whether the operation completed successfully
            details: Additional details about the operation completion
        """
        if self.current_operation and self.operation_start_time:
            import time
            duration = time.time() - self.operation_start_time
            
            status = "completed" if success else "failed"
            message = f"{self.current_operation} {status} in {duration:.2f}s"
            
            if details:
                message += f" - {details}"
            
            if success:
                self.info(message)
            else:
                self.error(message)
            
            self.current_operation = None
            self.operation_start_time = None
    
    def progress(self, current: int, total: int, message: str = "") -> None:
        """Log progress information.
        
        Args:
            current: Current progress count
            total: Total expected count
            message: Optional progress message
        """
        if not self.enable_progress:
            return
        
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"Progress: {current}/{total} ({percentage:.1f}%)"
        
        if message:
            progress_msg += f" - {message}"
        
        # Only log progress at certain intervals to avoid spam
        if current == 1 or current == total or current % max(1, total // 20) == 0:
            self.info(progress_msg)
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "Metrics") -> None:
        """Log metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for the log message
        """
        self.info(f"{prefix}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration in a readable format.
        
        Args:
            config: Configuration dictionary to log
        """
        self.info("Configuration:")
        self._log_dict_recursive(config, indent=2)
    
    def _log_dict_recursive(self, d: Dict[str, Any], indent: int = 0) -> None:
        """Recursively log dictionary contents.
        
        Args:
            d: Dictionary to log
            indent: Current indentation level
        """
        for key, value in d.items():
            prefix = " " * indent
            if isinstance(value, dict):
                self.info(f"{prefix}{key}:")
                self._log_dict_recursive(value, indent + 2)
            elif isinstance(value, list):
                self.info(f"{prefix}{key}: [{len(value)} items]")
            else:
                self.info(f"{prefix}{key}: {value}")
    
    def set_verbosity(self, level: int) -> None:
        """Change the logging verbosity level.
        
        Args:
            level: New logging level
        """
        self.console_level = level
        
        # Update console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
                
                # Update formatter based on new level
                if level <= logging.DEBUG:
                    handler.setFormatter(self.detailed_formatter)
                elif level <= logging.INFO:
                    handler.setFormatter(self.standard_formatter)
                else:
                    handler.setFormatter(self.simple_formatter)


class ConfigLoader:
    """Utility for loading configuration files."""
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logging.info(f"Loaded configuration from {filepath}")
            return config
            
        except Exception as e:
            logging.error(f"Failed to load configuration from {filepath}: {e}")
            raise
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logging.info(f"Loaded configuration from {filepath}")
            return config
            
        except Exception as e:
            logging.error(f"Failed to load configuration from {filepath}: {e}")
            raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any], 
                       required_keys: List[str]) -> Dict[str, Any]:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            required_keys: List of required keys
            
        Returns:
            Validation results
        """
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        return {
            'valid': len(missing_keys) == 0,
            'missing_keys': missing_keys,
            'provided_keys': list(config.keys())
        }


class FileManager:
    """Utility for managing files and directories."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def clean_directory(path: Union[str, Path], 
                       pattern: str = "*",
                       keep_subdirs: bool = True) -> int:
        """Clean files from directory.
        
        Args:
            path: Directory path
            pattern: File pattern to match
            keep_subdirs: Whether to keep subdirectories
            
        Returns:
            Number of files removed
        """
        path = Path(path)
        
        if not path.exists():
            return 0
        
        removed_count = 0
        
        for item in path.glob(pattern):
            if item.is_file():
                item.unlink()
                removed_count += 1
            elif item.is_dir() and not keep_subdirs:
                import shutil
                shutil.rmtree(item)
                removed_count += 1
        
        logging.info(f"Cleaned {removed_count} items from {path}")
        return removed_count
    
    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary containing file information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        
        return {
            'exists': True,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'is_file': filepath.is_file(),
            'is_dir': filepath.is_dir(),
            'extension': filepath.suffix,
            'name': filepath.name,
            'parent': str(filepath.parent)
        }
    
    @staticmethod
    def backup_file(filepath: Union[str, Path], 
                   backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create a backup of a file.
        
        Args:
            filepath: Path to file to backup
            backup_dir: Directory for backup (optional)
            
        Returns:
            Path to backup file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if backup_dir:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{filepath.stem}_backup_{int(datetime.now().timestamp())}{filepath.suffix}"
        else:
            backup_path = filepath.parent / f"{filepath.stem}_backup_{int(datetime.now().timestamp())}{filepath.suffix}"
        
        import shutil
        shutil.copy2(filepath, backup_path)
        
        logging.info(f"Created backup: {backup_path}")
        return backup_path