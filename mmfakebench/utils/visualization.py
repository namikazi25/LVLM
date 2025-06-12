"""Visualization utilities for benchmark results.

This module provides comprehensive visualization tools for benchmark results,
including performance plots, comparison charts, and result analysis visualizations.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization dependencies not available. Install matplotlib, seaborn, numpy, pandas for full functionality.")


class ResultVisualizer:
    """Comprehensive visualization utility for benchmark results."""
    
    def __init__(self, output_dir: Union[str, Path] = "visualizations"):
        """Initialize the result visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if VISUALIZATION_AVAILABLE:
            # Set style for better-looking plots
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
            sns.set_palette("husl")
    
    def plot_performance_metrics(self, 
                               results: List[Dict[str, Any]], 
                               metrics: List[str] = None,
                               save_path: Optional[str] = None) -> Optional[str]:
        """Create performance metrics visualization.
        
        Args:
            results: List of result dictionaries
            metrics: List of metrics to plot (default: ['accuracy', 'precision', 'recall', 'f1'])
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available. Install required dependencies.")
            return None
        
        if not results:
            logging.warning("No results provided for visualization.")
            return None
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract metrics from results
        metric_values = defaultdict(list)
        labels = []
        
        for i, result in enumerate(results):
            labels.append(f"Run {i+1}")
            for metric in metrics:
                value = result.get('metrics', {}).get(metric, 0)
                metric_values[metric].append(value)
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            if i < len(axes):
                ax = axes[i]
                values = metric_values[metric]
                
                ax.bar(labels, values, alpha=0.7)
                ax.set_title(f'{metric.title()} Across Runs')
                ax.set_ylabel(metric.title())
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for j, v in enumerate(values):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Performance metrics plot saved to {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, 
                            y_true: List[str], 
                            y_pred: List[str],
                            normalize: bool = False,
                            save_path: Optional[str] = None) -> Optional[str]:
        """Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available. Install required dependencies.")
            return None
        
        from sklearn.metrics import confusion_matrix
        
        # Convert labels to binary
        y_true_binary = [1 if label in ['fake', 'misinformation', 1, True] else 0 for label in y_true]
        y_pred_binary = [1 if label in ['fake', 'misinformation', 1, True] else 0 for label in y_pred]
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                   cmap='Blues', cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = self.output_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Confusion matrix plot saved to {save_path}")
        return str(save_path)
    
    def plot_cost_analysis(self, 
                         results: List[Dict[str, Any]],
                         save_path: Optional[str] = None) -> Optional[str]:
        """Create cost analysis visualization.
        
        Args:
            results: List of result dictionaries with cost information
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available. Install required dependencies.")
            return None
        
        # Extract cost data
        costs = []
        accuracies = []
        labels = []
        
        for i, result in enumerate(results):
            cost = result.get('cost', {}).get('total_cost_usd', 0)
            accuracy = result.get('metrics', {}).get('accuracy', 0)
            
            costs.append(cost)
            accuracies.append(accuracy)
            labels.append(f"Run {i+1}")
        
        if not costs or all(c == 0 for c in costs):
            logging.warning("No cost data available for visualization.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost vs Accuracy scatter plot
        ax1.scatter(costs, accuracies, alpha=0.7, s=100)
        ax1.set_xlabel('Total Cost (USD)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Cost vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, label in enumerate(labels):
            ax1.annotate(label, (costs[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Cost efficiency bar chart
        efficiency = [acc / max(cost, 0.001) for acc, cost in zip(accuracies, costs)]
        ax2.bar(labels, efficiency, alpha=0.7)
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Accuracy per Dollar')
        ax2.set_title('Cost Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Cost analysis plot saved to {save_path}")
        return str(save_path)
    
    def plot_comparison(self, 
                      results_list: List[List[Dict[str, Any]]], 
                      labels: List[str],
                      metric: str = 'accuracy',
                      save_path: Optional[str] = None) -> Optional[str]:
        """Create comparison visualization between different result sets.
        
        Args:
            results_list: List of result lists to compare
            labels: Labels for each result set
            metric: Metric to compare
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available. Install required dependencies.")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Extract metric values for each result set
        data_for_plot = []
        
        for results, label in zip(results_list, labels):
            values = []
            for result in results:
                value = result.get('metrics', {}).get(metric, 0)
                values.append(value)
            data_for_plot.append(values)
        
        # Create box plot
        plt.boxplot(data_for_plot, labels=labels)
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} Comparison Across Different Configurations')
        plt.grid(True, alpha=0.3)
        
        # Add mean values as text
        for i, values in enumerate(data_for_plot):
            if values:
                mean_val = np.mean(values)
                plt.text(i+1, mean_val, f'μ={mean_val:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Comparison plot saved to {save_path}")
        return str(save_path)
    
    def create_dashboard(self, 
                        results: List[Dict[str, Any]],
                        save_path: Optional[str] = None) -> Optional[str]:
        """Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            results: List of result dictionaries
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard or None if visualization not available
        """
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available. Install required dependencies.")
            return None
        
        if not results:
            logging.warning("No results provided for dashboard creation.")
            return None
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance metrics over time
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = [r.get('metrics', {}).get(metric, 0) for r in results]
            plt.plot(range(len(values)), values, marker='o', label=metric)
        plt.title('Performance Metrics Over Runs')
        plt.xlabel('Run Number')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Cost analysis
        ax2 = plt.subplot(3, 3, 2)
        costs = [r.get('cost', {}).get('total_cost_usd', 0) for r in results]
        if any(c > 0 for c in costs):
            plt.bar(range(len(costs)), costs)
            plt.title('Cost per Run')
            plt.xlabel('Run Number')
            plt.ylabel('Cost (USD)')
        else:
            plt.text(0.5, 0.5, 'No cost data available', ha='center', va='center', transform=ax2.transAxes)
            plt.title('Cost Analysis')
        
        # 3. Processing time
        ax3 = plt.subplot(3, 3, 3)
        times = [r.get('duration', 0) for r in results]
        if any(t > 0 for t in times):
            plt.plot(range(len(times)), times, marker='s', color='orange')
            plt.title('Processing Time per Run')
            plt.xlabel('Run Number')
            plt.ylabel('Time (seconds)')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax3.transAxes)
            plt.title('Processing Time')
        
        # 4. Accuracy distribution
        ax4 = plt.subplot(3, 3, 4)
        accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in results]
        if accuracies:
            plt.hist(accuracies, bins=10, alpha=0.7, edgecolor='black')
            plt.title('Accuracy Distribution')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
        
        # 5. Model comparison (if multiple models)
        ax5 = plt.subplot(3, 3, 5)
        model_performance = defaultdict(list)
        for result in results:
            model = result.get('config', {}).get('model', {}).get('name', 'Unknown')
            accuracy = result.get('metrics', {}).get('accuracy', 0)
            model_performance[model].append(accuracy)
        
        if len(model_performance) > 1:
            models = list(model_performance.keys())
            avg_performance = [np.mean(model_performance[model]) for model in models]
            plt.bar(models, avg_performance)
            plt.title('Average Performance by Model')
            plt.ylabel('Average Accuracy')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Single model used', ha='center', va='center', transform=ax5.transAxes)
            plt.title('Model Comparison')
        
        # 6. Dataset statistics
        ax6 = plt.subplot(3, 3, 6)
        dataset_sizes = [r.get('dataset_info', {}).get('size', 0) for r in results]
        if any(s > 0 for s in dataset_sizes):
            plt.scatter(range(len(dataset_sizes)), dataset_sizes, alpha=0.7)
            plt.title('Dataset Size per Run')
            plt.xlabel('Run Number')
            plt.ylabel('Dataset Size')
        else:
            plt.text(0.5, 0.5, 'No dataset size data', ha='center', va='center', transform=ax6.transAxes)
            plt.title('Dataset Statistics')
        
        # 7. Error analysis
        ax7 = plt.subplot(3, 3, 7)
        error_counts = [r.get('errors', {}).get('count', 0) for r in results]
        if any(e > 0 for e in error_counts):
            plt.bar(range(len(error_counts)), error_counts, color='red', alpha=0.7)
            plt.title('Error Count per Run')
            plt.xlabel('Run Number')
            plt.ylabel('Error Count')
        else:
            plt.text(0.5, 0.5, 'No errors recorded', ha='center', va='center', transform=ax7.transAxes)
            plt.title('Error Analysis')
        
        # 8. Summary statistics
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Calculate summary stats
        all_accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in results]
        all_costs = [r.get('cost', {}).get('total_cost_usd', 0) for r in results]
        
        summary_text = f"""
        SUMMARY STATISTICS
        
        Total Runs: {len(results)}
        Avg Accuracy: {np.mean(all_accuracies):.3f}
        Best Accuracy: {max(all_accuracies):.3f}
        Worst Accuracy: {min(all_accuracies):.3f}
        
        Total Cost: ${sum(all_costs):.4f}
        Avg Cost/Run: ${np.mean(all_costs):.4f}
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 9. Performance trend
        ax9 = plt.subplot(3, 3, 9)
        if len(results) > 1:
            # Calculate moving average
            window_size = min(3, len(all_accuracies))
            moving_avg = []
            for i in range(len(all_accuracies)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(all_accuracies[start_idx:i+1]))
            
            plt.plot(range(len(all_accuracies)), all_accuracies, 'o-', alpha=0.7, label='Actual')
            plt.plot(range(len(moving_avg)), moving_avg, 's-', alpha=0.7, label=f'Moving Avg ({window_size})')
            plt.title('Performance Trend')
            plt.xlabel('Run Number')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Need multiple runs for trend', ha='center', va='center', transform=ax9.transAxes)
            plt.title('Performance Trend')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Dashboard saved to {save_path}")
        return str(save_path)