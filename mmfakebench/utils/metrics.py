"""Metrics and Evaluation Utilities for MMFakeBench.

This module provides comprehensive evaluation metrics, confusion matrix utilities,
and performance analysis tools for misinformation detection benchmarks.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter, defaultdict

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations for basic metrics
    def accuracy_score(y_true, y_pred):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0.0
    
    def precision_score(y_true, y_pred, average='binary', pos_label=None, zero_division=0):
        if average == 'binary':
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            return tp / (tp + fp) if (tp + fp) > 0 else zero_division
        return zero_division
    
    def recall_score(y_true, y_pred, average='binary', pos_label=None, zero_division=0):
        if average == 'binary':
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            return tp / (tp + fn) if (tp + fn) > 0 else zero_division
        return zero_division
    
    def f1_score(y_true, y_pred, average='binary', pos_label=None, zero_division=0):
        prec = precision_score(y_true, y_pred, average, pos_label, zero_division)
        rec = recall_score(y_true, y_pred, average, pos_label, zero_division)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else zero_division
    
    def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
        # Simple 2x2 confusion matrix for binary classification
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return [[tn, fp], [fn, tp]]
    
    def classification_report(y_true, y_pred, output_dict=False, zero_division=0):
        if output_dict:
            return {"accuracy": accuracy_score(y_true, y_pred)}
        return "Classification report requires sklearn"
    
    def roc_auc_score(y_true, y_score):
        return 0.5  # Random baseline
    
    def precision_recall_curve(y_true, y_score):
        return [0, 1], [0, 1], [0.5]
    
    def roc_curve(y_true, y_score):
        return [0, 1], [0, 1], [0.5]
    
    # Mock matplotlib and seaborn
    class MockPlt:
        def figure(self, figsize=None): pass
        def title(self, title): pass
        def ylabel(self, label): pass
        def xlabel(self, label): pass
        def tight_layout(self): pass
        def savefig(self, path, **kwargs): pass
        def close(self): pass
    
    class MockSns:
        def heatmap(self, data, **kwargs): pass
    
    plt = MockPlt()
    sns = MockSns()
    
    # Mock numpy and pandas
    class MockNumpy:
        def array(self, data): return data
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): return 0
        def sum(self, data): return sum(data)
        def where(self, condition): return [i for i, x in enumerate(condition) if x]
        def linspace(self, start, stop, num): return [start + i * (stop - start) / (num - 1) for i in range(num)]
    
    class MockPandas:
        def DataFrame(self, data): return data
        def read_csv(self, path): return {}
    
    np = MockNumpy()
    pd = MockPandas()


class MetricsCalculator:
    """Comprehensive metrics calculator for misinformation detection evaluation."""
    
    def __init__(self, class_labels: Optional[List[str]] = None):
        """Initialize the metrics calculator.
        
        Args:
            class_labels: List of class labels for classification
        """
        self.class_labels = class_labels or ['real', 'fake']
        self.binary_mode = len(self.class_labels) == 2
        
    def calculate_basic_metrics(self, 
                              y_true: List[str], 
                              y_pred: List[str]) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing basic metrics
        """
        # Clean predictions (convert to binary if needed)
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        
        if SKLEARN_AVAILABLE:
            metrics = {
                'accuracy': accuracy_score(y_true_clean, y_pred_clean),
                'precision': precision_score(y_true_clean, y_pred_clean, zero_division=0),
                'recall': recall_score(y_true_clean, y_pred_clean, zero_division=0),
                'f1_score': f1_score(y_true_clean, y_pred_clean, zero_division=0),
                'support': len(y_true_clean)
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_true_clean, y_pred_clean),
                'precision': precision_score(y_true_clean, y_pred_clean, zero_division=0),
                'recall': recall_score(y_true_clean, y_pred_clean, zero_division=0),
                'f1_score': f1_score(y_true_clean, y_pred_clean, zero_division=0),
                'support': len(y_true_clean)
            }
        

        
        return metrics
    
    def calculate_confidence_metrics(self, 
                                   y_true: List[str], 
                                   y_pred: List[str],
                                   confidence_scores: List[float]) -> Dict[str, float]:
        """Calculate confidence-based metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Dictionary containing confidence metrics
        """
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        
        # Calculate accuracy at different confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_metrics = {}
        
        for threshold in thresholds:
            high_conf_mask = np.array(confidence_scores) >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = accuracy_score(
                    np.array(y_true_clean)[high_conf_mask],
                    np.array(y_pred_clean)[high_conf_mask]
                )
                confidence_metrics[f'accuracy_conf_{threshold}'] = high_conf_acc
                confidence_metrics[f'coverage_conf_{threshold}'] = np.mean(high_conf_mask)
            else:
                confidence_metrics[f'accuracy_conf_{threshold}'] = 0.0
                confidence_metrics[f'coverage_conf_{threshold}'] = 0.0
        
        # Calculate calibration metrics
        confidence_metrics.update(self._calculate_calibration_metrics(y_true_clean, y_pred_clean, confidence_scores))
        
        return confidence_metrics
    
    def generate_confusion_matrix(self, 
                                y_true: List[str], 
                                y_pred: List[str],
                                normalize: Optional[str] = None,
                                save_path: Optional[str] = None) -> List[List[int]]:
        """Generate and optionally save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            save_path: Path to save the confusion matrix plot
            
        Returns:
            Confusion matrix as list of lists
        """
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        
        cm = confusion_matrix(y_true_clean, y_pred_clean)
        
        if save_path and SKLEARN_AVAILABLE:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                       cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return cm
    
    def generate_classification_report(self, 
                                     y_true: List[str], 
                                     y_pred: List[str]) -> Dict[str, Any]:
        """Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as dictionary
        """
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        
        report = classification_report(y_true_clean, y_pred_clean, output_dict=True, zero_division=0)
        
        # Add custom metrics
        report['custom_metrics'] = {
            'total_samples': len(y_true_clean),
            'class_distribution': dict(Counter(y_true_clean)),
            'prediction_distribution': dict(Counter(y_pred_clean))
        }
        
        return report
    
    def calculate_cost_metrics(self, 
                             y_true: List[str], 
                             y_pred: List[str],
                             api_costs: List[float],
                             processing_times: List[float]) -> Dict[str, float]:
        """Calculate cost and efficiency metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            api_costs: API costs per sample
            processing_times: Processing times per sample
            
        Returns:
            Dictionary containing cost metrics
        """
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        
        total_cost = sum(api_costs)
        total_time = sum(processing_times)
        avg_cost = sum(api_costs) / len(api_costs) if api_costs else 0
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        correct_predictions = sum(1 for t, p in zip(y_true_clean, y_pred_clean) if t == p)
        
        return {
            'total_cost_usd': total_cost,
            'average_cost_per_sample': avg_cost,
            'cost_per_correct_prediction': total_cost / max(1, correct_predictions),
            'accuracy_per_dollar': accuracy / max(0.001, total_cost),
            'total_processing_time': total_time,
            'average_time_per_sample': avg_time,
            'throughput_samples_per_second': len(y_true_clean) / max(0.001, total_time)
        }
    
    def analyze_error_patterns(self, 
                             y_true: List[str], 
                             y_pred: List[str],
                             metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns and failure modes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metadata: Metadata for each sample (e.g., image type, claim category)
            
        Returns:
            Dictionary containing error analysis
        """
        y_pred_clean = [1 if pred in ['fake', 'misinformation', 1, True] else 0 
                       for pred in y_pred]
        y_true_clean = [1 if true in ['fake', 'misinformation', 1, True] else 0 
                       for true in y_true]
        
        # Identify errors
        errors = np.array(y_true_clean) != np.array(y_pred_clean)
        error_indices = np.where(errors)[0]
        
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_true_clean),
            'error_patterns': defaultdict(int)
        }
        
        # Analyze error patterns by metadata
        for idx in error_indices:
            if idx < len(metadata):
                meta = metadata[idx]
                for key, value in meta.items():
                    if isinstance(value, (str, int, float)):
                        error_analysis['error_patterns'][f'{key}_{value}'] += 1
        
        # Convert defaultdict to regular dict
        error_analysis['error_patterns'] = dict(error_analysis['error_patterns'])
        
        return error_analysis
    
    def compare_models(self, 
                      results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model results.
        
        Args:
            results: Dictionary mapping model names to their results
            
        Returns:
            DataFrame comparing model performance
        """
        comparison_data = []
        
        for model_name, result in results.items():
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            metrics = self.calculate_basic_metrics(y_true, y_pred)
            
            # Add cost metrics if available
            if 'api_costs' in result and 'processing_times' in result:
                cost_metrics = self.calculate_cost_metrics(
                    y_true, y_pred, result['api_costs'], result['processing_times']
                )
                metrics.update(cost_metrics)
            
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data).set_index('model')
    

    
    def _calculate_calibration_metrics(self, 
                                     y_true: List[str], 
                                     y_pred: List[str],
                                     confidence_scores: List[float]) -> Dict[str, float]:
        """Calculate calibration metrics for confidence scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidence_scores: Confidence scores
            
        Returns:
            Dictionary containing calibration metrics
        """
        # Convert to binary for calibration calculation
        if self.binary_mode:
            y_true_binary = [1 if label == 'fake' else 0 for label in y_true]
            y_pred_binary = [1 if label == 'fake' else 0 for label in y_pred]
            
            # Calculate Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_true_binary, confidence_scores)
            
            # Calculate Brier Score
            brier_score = np.mean((np.array(confidence_scores) - np.array(y_true_binary)) ** 2)
            
            return {
                'expected_calibration_error': ece,
                'brier_score': brier_score,
                'average_confidence': np.mean(confidence_scores),
                'confidence_std': np.std(confidence_scores)
            }
        
        return {
            'average_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores)
        }
    
    def _calculate_ece(self, 
                      y_true: List[int], 
                      confidence_scores: List[float], 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error.
        
        Args:
            y_true: Binary true labels
            confidence_scores: Confidence scores
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(y_true)[in_bin].mean()
                avg_confidence_in_bin = np.array(confidence_scores)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class BenchmarkAnalyzer:
    """High-level analyzer for benchmark results."""
    
    def __init__(self, results_dir: str):
        """Initialize the benchmark analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.metrics_calculator = MetricsCalculator()
    
    def load_results(self, filename: str) -> pd.DataFrame:
        """Load benchmark results from CSV file.
        
        Args:
            filename: Name of the results file
            
        Returns:
            DataFrame containing results
        """
        filepath = self.results_dir / filename
        return pd.read_csv(filepath)
    
    def generate_comprehensive_report(self, 
                                    results_df: pd.DataFrame,
                                    output_dir: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Args:
            results_df: DataFrame containing benchmark results
            output_dir: Directory to save analysis outputs
            
        Returns:
            Dictionary containing analysis summary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract required columns
        y_true = results_df['true_label'].tolist()
        y_pred = results_df['predicted_label'].tolist()
        
        # Calculate basic metrics
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(y_true, y_pred)
        
        # Generate confusion matrix
        cm_path = output_path / 'confusion_matrix.png'
        confusion_matrix = self.metrics_calculator.generate_confusion_matrix(
            y_true, y_pred, normalize='true', save_path=str(cm_path)
        )
        
        # Generate classification report
        classification_report = self.metrics_calculator.generate_classification_report(y_true, y_pred)
        
        # Calculate confidence metrics if available
        confidence_metrics = {}
        if 'confidence_score' in results_df.columns:
            confidence_scores = results_df['confidence_score'].tolist()
            confidence_metrics = self.metrics_calculator.calculate_confidence_metrics(
                y_true, y_pred, confidence_scores
            )
        
        # Calculate cost metrics if available
        cost_metrics = {}
        if 'api_cost' in results_df.columns and 'processing_time' in results_df.columns:
            api_costs = results_df['api_cost'].tolist()
            processing_times = results_df['processing_time'].tolist()
            cost_metrics = self.metrics_calculator.calculate_cost_metrics(
                y_true, y_pred, api_costs, processing_times
            )
        
        # Analyze error patterns
        metadata = results_df.to_dict('records')
        error_analysis = self.metrics_calculator.analyze_error_patterns(y_true, y_pred, metadata)
        
        # Compile comprehensive report
        report = {
            'basic_metrics': basic_metrics,
            'classification_report': classification_report,
            'confidence_metrics': confidence_metrics,
            'cost_metrics': cost_metrics,
            'error_analysis': error_analysis,
            'confusion_matrix_path': str(cm_path)
        }
        
        # Save report as JSON
        import json
        report_path = output_path / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report