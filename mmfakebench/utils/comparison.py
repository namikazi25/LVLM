"""Result comparison utilities for benchmark analysis.

This module provides tools for comparing benchmark results across different runs,
configurations, models, and datasets.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import statistics

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Some comparison features will be limited.")


class ResultComparator:
    """Utility for comparing benchmark results across different runs and configurations."""
    
    def __init__(self, output_dir: Union[str, Path] = "comparisons"):
        """Initialize the result comparator.
        
        Args:
            output_dir: Directory to save comparison outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_runs(self, 
                    results_list: List[List[Dict[str, Any]]], 
                    run_labels: List[str],
                    metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple benchmark runs.
        
        Args:
            results_list: List of result lists to compare
            run_labels: Labels for each run
            metrics: Metrics to compare (default: common metrics)
            
        Returns:
            Comparison results dictionary
        """
        if len(results_list) != len(run_labels):
            raise ValueError("Number of result lists must match number of labels")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        comparison = {
            'run_labels': run_labels,
            'metrics_compared': metrics,
            'comparison_timestamp': datetime.now().isoformat(),
            'run_statistics': {},
            'metric_comparisons': {},
            'best_performers': {},
            'statistical_significance': {}
        }
        
        # Calculate statistics for each run
        for i, (results, label) in enumerate(zip(results_list, run_labels)):
            run_stats = self._calculate_run_statistics(results, metrics)
            comparison['run_statistics'][label] = run_stats
        
        # Compare metrics across runs
        for metric in metrics:
            metric_comparison = self._compare_metric_across_runs(
                results_list, run_labels, metric
            )
            comparison['metric_comparisons'][metric] = metric_comparison
            
            # Find best performer for this metric
            best_run = max(metric_comparison['mean_values'].items(), 
                          key=lambda x: x[1])
            comparison['best_performers'][metric] = {
                'run': best_run[0],
                'value': best_run[1]
            }
        
        # Calculate statistical significance if possible
        if len(results_list) >= 2:
            comparison['statistical_significance'] = self._calculate_statistical_significance(
                results_list, run_labels, metrics
            )
        
        return comparison
    
    def compare_models(self, 
                      results: List[Dict[str, Any]],
                      group_by_model: bool = True) -> Dict[str, Any]:
        """Compare performance across different models.
        
        Args:
            results: List of results with model information
            group_by_model: Whether to group by model name
            
        Returns:
            Model comparison results
        """
        model_groups = defaultdict(list)
        
        for result in results:
            model_info = result.get('model_info', {})
            if group_by_model:
                model_key = model_info.get('name', 'Unknown')
            else:
                # Group by model name + version
                name = model_info.get('name', 'Unknown')
                version = model_info.get('version', 'Unknown')
                model_key = f"{name}_{version}"
            
            model_groups[model_key].append(result)
        
        comparison = {
            'models_compared': list(model_groups.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'model_statistics': {},
            'performance_ranking': {},
            'cost_efficiency': {}
        }
        
        # Calculate statistics for each model
        for model_name, model_results in model_groups.items():
            stats = self._calculate_run_statistics(model_results)
            comparison['model_statistics'][model_name] = stats
            
            # Calculate cost efficiency
            total_cost = sum(r.get('api_cost', 0) for r in model_results)
            avg_accuracy = stats.get('metrics', {}).get('accuracy', {}).get('mean', 0)
            
            # Prevent division by zero
            safe_accuracy = max(avg_accuracy, 0.001)
            safe_cost = max(total_cost, 0.001)
            
            comparison['cost_efficiency'][model_name] = {
                'total_cost': total_cost,
                'average_accuracy': avg_accuracy,
                'cost_per_accuracy': total_cost / safe_accuracy if avg_accuracy > 0 else float('inf'),
                'accuracy_per_dollar': avg_accuracy / safe_cost if total_cost > 0 else 0
            }
        
        # Rank models by different criteria
        comparison['performance_ranking'] = self._rank_models(
            comparison['model_statistics']
        )
        
        return comparison
    
    def compare_datasets(self, 
                        results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance across different datasets.
        
        Args:
            results: List of results with dataset information
            
        Returns:
            Dataset comparison results
        """
        dataset_groups = defaultdict(list)
        
        for result in results:
            dataset_info = result.get('dataset_info', {})
            dataset_key = dataset_info.get('name', 'Unknown')
            dataset_groups[dataset_key].append(result)
        
        comparison = {
            'datasets_compared': list(dataset_groups.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'dataset_statistics': {},
            'difficulty_analysis': {},
            'performance_correlation': {}
        }
        
        # Calculate statistics for each dataset
        for dataset_name, dataset_results in dataset_groups.items():
            stats = self._calculate_run_statistics(dataset_results)
            comparison['dataset_statistics'][dataset_name] = stats
            
            # Analyze dataset difficulty
            accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in dataset_results]
            comparison['difficulty_analysis'][dataset_name] = {
                'average_accuracy': statistics.mean(accuracies) if accuracies else 0,
                'accuracy_std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                'difficulty_score': 1 - statistics.mean(accuracies) if accuracies else 1,
                'consistency_score': 1 - (statistics.stdev(accuracies) if len(accuracies) > 1 else 0)
            }
        
        return comparison
    
    def generate_comparison_report(self, 
                                 comparison_data: Dict[str, Any],
                                 report_type: str = "runs") -> str:
        """Generate a human-readable comparison report.
        
        Args:
            comparison_data: Comparison data from compare_* methods
            report_type: Type of comparison ("runs", "models", "datasets")
            
        Returns:
            Formatted comparison report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"BENCHMARK COMPARISON REPORT - {report_type.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if report_type == "runs":
            report_lines.extend(self._generate_runs_report(comparison_data))
        elif report_type == "models":
            report_lines.extend(self._generate_models_report(comparison_data))
        elif report_type == "datasets":
            report_lines.extend(self._generate_datasets_report(comparison_data))
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_comparison(self, 
                       comparison_data: Dict[str, Any],
                       filename: Optional[str] = None,
                       format: str = "json") -> Path:
        """Save comparison results to file.
        
        Args:
            comparison_data: Comparison data to save
            filename: Output filename (auto-generated if None)
            format: Output format ("json" or "csv")
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comparison_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        elif format == "csv" and PANDAS_AVAILABLE:
            # Convert to DataFrame for CSV export
            df = self._comparison_to_dataframe(comparison_data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Comparison results saved to {filepath}")
        return filepath
    
    def _calculate_run_statistics(self, 
                                results: List[Dict[str, Any]], 
                                metrics: List[str] = None) -> Dict[str, Any]:
        """Calculate statistics for a single run.
        
        Args:
            results: List of results for the run
            metrics: Metrics to calculate statistics for
            
        Returns:
            Statistics dictionary
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        stats = {
            'total_samples': len(results),
            'metrics': {}
        }
        
        for metric in metrics:
            values = [r.get('metrics', {}).get(metric, 0) for r in results]
            values = [v for v in values if v is not None]  # Filter None values
            
            if values:
                stats['metrics'][metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                stats['metrics'][metric] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        # Calculate cost statistics
        costs = [r.get('api_cost', 0) for r in results]
        costs = [c for c in costs if c > 0]
        
        if costs:
            stats['cost'] = {
                'total': sum(costs),
                'mean': statistics.mean(costs),
                'median': statistics.median(costs),
                'std': statistics.stdev(costs) if len(costs) > 1 else 0
            }
        
        # Calculate timing statistics
        times = [r.get('processing_time', 0) for r in results]
        times = [t for t in times if t > 0]
        
        if times:
            stats['timing'] = {
                'total': sum(times),
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0
            }
        
        return stats
    
    def _compare_metric_across_runs(self, 
                                  results_list: List[List[Dict[str, Any]]], 
                                  run_labels: List[str],
                                  metric: str) -> Dict[str, Any]:
        """Compare a specific metric across multiple runs.
        
        Args:
            results_list: List of result lists
            run_labels: Labels for each run
            metric: Metric to compare
            
        Returns:
            Metric comparison data
        """
        comparison = {
            'metric': metric,
            'mean_values': {},
            'std_values': {},
            'improvement': {},
            'relative_performance': {}
        }
        
        # Calculate mean and std for each run
        for results, label in zip(results_list, run_labels):
            values = [r.get('metrics', {}).get(metric, 0) for r in results]
            values = [v for v in values if v is not None]
            
            if values:
                comparison['mean_values'][label] = statistics.mean(values)
                comparison['std_values'][label] = statistics.stdev(values) if len(values) > 1 else 0
            else:
                comparison['mean_values'][label] = 0
                comparison['std_values'][label] = 0
        
        # Calculate improvements relative to first run
        if run_labels:
            baseline = comparison['mean_values'][run_labels[0]]
            
            for label in run_labels[1:]:
                current_value = comparison['mean_values'][label]
                if baseline > 0:
                    improvement = ((current_value - baseline) / baseline) * 100
                else:
                    improvement = 0 if current_value == 0 else float('inf')
                comparison['improvement'][label] = improvement
        
        # Calculate relative performance (percentage of best)
        if comparison['mean_values']:
            best_value = max(comparison['mean_values'].values())
            
            for label, value in comparison['mean_values'].items():
                if best_value > 0:
                    relative_perf = (value / best_value) * 100
                else:
                    relative_perf = 0
                comparison['relative_performance'][label] = relative_perf
        
        return comparison
    
    def _calculate_statistical_significance(self, 
                                          results_list: List[List[Dict[str, Any]]], 
                                          run_labels: List[str],
                                          metrics: List[str]) -> Dict[str, Any]:
        """Calculate statistical significance between runs.
        
        Args:
            results_list: List of result lists
            run_labels: Labels for each run
            metrics: Metrics to test
            
        Returns:
            Statistical significance results
        """
        significance = {
            'method': 'basic_comparison',  # Could be enhanced with proper statistical tests
            'comparisons': {}
        }
        
        # Simple pairwise comparisons
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                pair_key = f"{run_labels[i]}_vs_{run_labels[j]}"
                significance['comparisons'][pair_key] = {}
                
                for metric in metrics:
                    values_i = [r.get('metrics', {}).get(metric, 0) for r in results_list[i]]
                    values_j = [r.get('metrics', {}).get(metric, 0) for r in results_list[j]]
                    
                    values_i = [v for v in values_i if v is not None]
                    values_j = [v for v in values_j if v is not None]
                    
                    if values_i and values_j:
                        mean_i = statistics.mean(values_i)
                        mean_j = statistics.mean(values_j)
                        
                        # Simple effect size calculation
                        pooled_std = statistics.stdev(values_i + values_j) if len(values_i + values_j) > 1 else 1
                        # Prevent division by zero
                        safe_pooled_std = max(pooled_std, 0.001)
                        effect_size = abs(mean_i - mean_j) / safe_pooled_std
                        
                        significance['comparisons'][pair_key][metric] = {
                            'mean_difference': mean_j - mean_i,
                            'effect_size': effect_size,
                            'practical_significance': effect_size > 0.2  # Cohen's d threshold
                        }
        
        return significance
    
    def _rank_models(self, model_statistics: Dict[str, Any]) -> Dict[str, List[str]]:
        """Rank models by different criteria.
        
        Args:
            model_statistics: Statistics for each model
            
        Returns:
            Rankings by different criteria
        """
        rankings = {}
        
        # Rank by accuracy
        accuracy_scores = {}
        for model, stats in model_statistics.items():
            accuracy_scores[model] = stats.get('metrics', {}).get('accuracy', {}).get('mean', 0)
        
        rankings['by_accuracy'] = sorted(accuracy_scores.keys(), 
                                       key=lambda x: accuracy_scores[x], 
                                       reverse=True)
        
        # Rank by F1 score
        f1_scores = {}
        for model, stats in model_statistics.items():
            f1_scores[model] = stats.get('metrics', {}).get('f1_score', {}).get('mean', 0)
        
        rankings['by_f1_score'] = sorted(f1_scores.keys(), 
                                       key=lambda x: f1_scores[x], 
                                       reverse=True)
        
        return rankings
    
    def _generate_runs_report(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate report section for run comparisons."""
        lines = []
        
        lines.append("RUN COMPARISON SUMMARY:")
        lines.append("-" * 40)
        
        run_stats = comparison_data.get('run_statistics', {})
        for run_label, stats in run_stats.items():
            lines.append(f"\n{run_label}:")
            metrics = stats.get('metrics', {})
            for metric, metric_stats in metrics.items():
                mean_val = metric_stats.get('mean', 0)
                std_val = metric_stats.get('std', 0)
                lines.append(f"  {metric}: {mean_val:.4f} (±{std_val:.4f})")
        
        # Best performers
        lines.append("\nBEST PERFORMERS:")
        lines.append("-" * 20)
        best_performers = comparison_data.get('best_performers', {})
        for metric, performer in best_performers.items():
            lines.append(f"{metric}: {performer['run']} ({performer['value']:.4f})")
        
        return lines
    
    def _generate_models_report(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate report section for model comparisons."""
        lines = []
        
        lines.append("MODEL COMPARISON SUMMARY:")
        lines.append("-" * 40)
        
        # Performance ranking
        rankings = comparison_data.get('performance_ranking', {})
        if 'by_accuracy' in rankings:
            lines.append("\nRanking by Accuracy:")
            for i, model in enumerate(rankings['by_accuracy'], 1):
                lines.append(f"  {i}. {model}")
        
        # Cost efficiency
        lines.append("\nCost Efficiency:")
        cost_efficiency = comparison_data.get('cost_efficiency', {})
        for model, efficiency in cost_efficiency.items():
            acc_per_dollar = efficiency.get('accuracy_per_dollar', 0)
            lines.append(f"  {model}: {acc_per_dollar:.4f} accuracy per dollar")
        
        return lines
    
    def _generate_datasets_report(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate report section for dataset comparisons."""
        lines = []
        
        lines.append("DATASET COMPARISON SUMMARY:")
        lines.append("-" * 40)
        
        difficulty_analysis = comparison_data.get('difficulty_analysis', {})
        for dataset, analysis in difficulty_analysis.items():
            lines.append(f"\n{dataset}:")
            lines.append(f"  Average Accuracy: {analysis.get('average_accuracy', 0):.4f}")
            lines.append(f"  Difficulty Score: {analysis.get('difficulty_score', 0):.4f}")
            lines.append(f"  Consistency Score: {analysis.get('consistency_score', 0):.4f}")
        
        return lines
    
    def _comparison_to_dataframe(self, comparison_data: Dict[str, Any]):
        """Convert comparison data to pandas DataFrame for CSV export."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for CSV export")
        
        # This is a simplified conversion - could be enhanced based on specific needs
        flattened_data = []
        
        # Extract run statistics if available
        if 'run_statistics' in comparison_data:
            for run_label, stats in comparison_data['run_statistics'].items():
                row = {'run': run_label}
                
                # Add metrics
                metrics = stats.get('metrics', {})
                for metric, metric_stats in metrics.items():
                    for stat_name, stat_value in metric_stats.items():
                        row[f'{metric}_{stat_name}'] = stat_value
                
                flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)