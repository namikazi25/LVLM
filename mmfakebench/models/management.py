#!/usr/bin/env python3
"""
Model Management Module

This module provides classes for managing model usage, including:
- Rate limiting
- Quota management
- Performance monitoring
- Model health checking
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Configuration for rate limiting."""
    
    def __init__(self,
                 requests_per_minute: int = 60,
                 tokens_per_minute: int = 10000,
                 requests_per_hour: int = 1000,
                 tokens_per_hour: int = 100000):
        """Initialize rate limit configuration.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            tokens_per_minute: Maximum number of tokens per minute
            requests_per_hour: Maximum number of requests per hour
            tokens_per_hour: Maximum number of tokens per hour
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_hour = requests_per_hour
        self.tokens_per_hour = tokens_per_hour


class QuotaConfig:
    """Configuration for quota management."""
    
    def __init__(self,
                 daily_cost_limit: float = 10.0,
                 monthly_cost_limit: float = 100.0,
                 daily_request_limit: int = 1000,
                 monthly_request_limit: int = 10000):
        """Initialize quota configuration.
        
        Args:
            daily_cost_limit: Maximum cost per day in USD
            monthly_cost_limit: Maximum cost per month in USD
            daily_request_limit: Maximum number of requests per day
            monthly_request_limit: Maximum number of requests per month
        """
        self.daily_cost_limit = daily_cost_limit
        self.monthly_cost_limit = monthly_cost_limit
        self.daily_request_limit = daily_request_limit
        self.monthly_request_limit = monthly_request_limit


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        """Initialize performance metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.error_counts: Dict[str, int] = {}
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_request_time = None
    
    def record_request(self,
                       model_name: str,
                       success: bool,
                       latency: float,
                       tokens_used: int,
                       cost: float,
                       error_type: Optional[str] = None):
        """Record metrics for a request.
        
        Args:
            model_name: Name of the model used
            success: Whether the request was successful
            latency: Request latency in seconds
            tokens_used: Number of tokens used
            cost: Cost of the request in USD
            error_type: Type of error if request failed
        """
        self.total_requests += 1
        self.total_latency += latency
        self.total_tokens += tokens_used
        self.total_cost += cost
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update model-specific metrics
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {
                'requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_latency': 0.0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'error_counts': {}
            }
        
        model_metrics = self.model_metrics[model_name]
        model_metrics['requests'] += 1
        model_metrics['total_latency'] += latency
        model_metrics['total_tokens'] += tokens_used
        model_metrics['total_cost'] += cost
        
        if success:
            model_metrics['successful_requests'] += 1
        else:
            model_metrics['failed_requests'] += 1
            if error_type:
                model_metrics['error_counts'][error_type] = model_metrics['error_counts'].get(error_type, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0,
            'average_latency': (self.total_latency / self.total_requests) if self.total_requests > 0 else 0,
            'average_tokens_per_request': (self.total_tokens / self.total_requests) if self.total_requests > 0 else 0,
            'average_cost_per_request': (self.total_cost / self.total_requests) if self.total_requests > 0 else 0,
            'error_counts': self.error_counts,
            'model_metrics': {}
        }
        
        # Add model-specific metrics
        for model_name, model_data in self.model_metrics.items():
            metrics['model_metrics'][model_name] = {
                'requests': model_data['requests'],
                'successful_requests': model_data['successful_requests'],
                'failed_requests': model_data['failed_requests'],
                'success_rate': (model_data['successful_requests'] / model_data['requests']) if model_data['requests'] > 0 else 0,
                'average_latency': (model_data['total_latency'] / model_data['requests']) if model_data['requests'] > 0 else 0,
                'average_tokens_per_request': (model_data['total_tokens'] / model_data['requests']) if model_data['requests'] > 0 else 0,
                'average_cost_per_request': (model_data['total_cost'] / model_data['requests']) if model_data['requests'] > 0 else 0,
                'error_counts': model_data['error_counts']
            }
        
        return metrics


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset rate limiter state."""
        self.minute_start_time = time.time()
        self.hour_start_time = time.time()
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.requests_this_hour = 0
        self.tokens_this_hour = 0
    
    def _update_windows(self):
        """Update time windows and reset counters if needed."""
        current_time = time.time()
        
        # Check if minute window has passed
        if current_time - self.minute_start_time >= 60:
            self.minute_start_time = current_time
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
        
        # Check if hour window has passed
        if current_time - self.hour_start_time >= 3600:
            self.hour_start_time = current_time
            self.requests_this_hour = 0
            self.tokens_this_hour = 0
    
    def can_make_request(self, tokens: int = 0) -> Tuple[bool, Optional[str]]:
        """Check if a request can be made within rate limits.
        
        Args:
            tokens: Number of tokens for the request
            
        Returns:
            Tuple of (can_proceed, reason)
            - can_proceed: True if request can proceed, False otherwise
            - reason: Reason for denial if can_proceed is False, None otherwise
        """
        self._update_windows()
        
        # Check minute limits
        if self.requests_this_minute >= self.config.requests_per_minute:
            return False, f"Exceeded requests per minute limit ({self.config.requests_per_minute})"
        
        if self.tokens_this_minute + tokens > self.config.tokens_per_minute:
            return False, f"Exceeded tokens per minute limit ({self.config.tokens_per_minute})"
        
        # Check hour limits
        if self.requests_this_hour >= self.config.requests_per_hour:
            return False, f"Exceeded requests per hour limit ({self.config.requests_per_hour})"
        
        if self.tokens_this_hour + tokens > self.config.tokens_per_hour:
            return False, f"Exceeded tokens per hour limit ({self.config.tokens_per_hour})"
        
        return True, None
    
    def record_request(self, tokens: int = 0):
        """Record a request.
        
        Args:
            tokens: Number of tokens used in the request
        """
        self._update_windows()
        
        self.requests_this_minute += 1
        self.tokens_this_minute += tokens
        self.requests_this_hour += 1
        self.tokens_this_hour += tokens
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status.
        
        Returns:
            Dictionary containing rate limiter status
        """
        self._update_windows()
        
        return {
            'requests_this_minute': self.requests_this_minute,
            'tokens_this_minute': self.tokens_this_minute,
            'requests_this_hour': self.requests_this_hour,
            'tokens_this_hour': self.tokens_this_hour,
            'minute_limit_remaining': self.config.requests_per_minute - self.requests_this_minute,
            'tokens_minute_limit_remaining': self.config.tokens_per_minute - self.tokens_this_minute,
            'hour_limit_remaining': self.config.requests_per_hour - self.requests_this_hour,
            'tokens_hour_limit_remaining': self.config.tokens_per_hour - self.tokens_this_hour,
            'minute_reset_in': int(60 - (time.time() - self.minute_start_time)),
            'hour_reset_in': int(3600 - (time.time() - self.hour_start_time))
        }


class QuotaManager:
    """Manager for API usage quotas."""
    
    def __init__(self, config: QuotaConfig):
        """Initialize quota manager.
        
        Args:
            config: Quota configuration
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset quota manager state."""
        self.daily_start_time = datetime.now()
        self.monthly_start_time = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.daily_cost_used = 0.0
        self.monthly_cost_used = 0.0
        self.daily_requests_used = 0
        self.monthly_requests_used = 0
    
    def _update_windows(self):
        """Update time windows and reset counters if needed."""
        current_time = datetime.now()
        
        # Check if day has changed
        if current_time.date() > self.daily_start_time.date():
            self.daily_start_time = current_time
            self.daily_cost_used = 0.0
            self.daily_requests_used = 0
        
        # Check if month has changed
        if current_time.month != self.monthly_start_time.month or current_time.year != self.monthly_start_time.year:
            self.monthly_start_time = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            self.monthly_cost_used = 0.0
            self.monthly_requests_used = 0
    
    def can_make_request(self, estimated_cost: float = 0.0) -> Tuple[bool, Optional[str]]:
        """Check if a request can be made within quotas.
        
        Args:
            estimated_cost: Estimated cost of the request in USD
            
        Returns:
            Tuple of (can_proceed, reason)
            - can_proceed: True if request can proceed, False otherwise
            - reason: Reason for denial if can_proceed is False, None otherwise
        """
        self._update_windows()
        
        # Check daily limits
        if self.daily_requests_used >= self.config.daily_request_limit:
            return False, f"Exceeded daily request limit ({self.config.daily_request_limit})"
        
        if self.daily_cost_used + estimated_cost > self.config.daily_cost_limit:
            return False, f"Exceeded daily cost limit (${self.config.daily_cost_limit:.2f})"
        
        # Check monthly limits
        if self.monthly_requests_used >= self.config.monthly_request_limit:
            return False, f"Exceeded monthly request limit ({self.config.monthly_request_limit})"
        
        if self.monthly_cost_used + estimated_cost > self.config.monthly_cost_limit:
            return False, f"Exceeded monthly cost limit (${self.config.monthly_cost_limit:.2f})"
        
        return True, None
    
    def record_usage(self, cost: float = 0.0):
        """Record usage.
        
        Args:
            cost: Cost of the request in USD
        """
        self._update_windows()
        
        self.daily_cost_used += cost
        self.monthly_cost_used += cost
        self.daily_requests_used += 1
        self.monthly_requests_used += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get quota manager status.
        
        Returns:
            Dictionary containing quota manager status
        """
        self._update_windows()
        
        # Calculate days left in month
        now = datetime.now()
        next_month = now.replace(day=28) + timedelta(days=4)  # This will always land in the next month
        next_month = next_month.replace(day=1)  # First day of next month
        days_in_month = (next_month - now.replace(day=1)).days
        days_left_in_month = days_in_month - now.day + 1
        
        return {
            'daily_cost_used': self.daily_cost_used,
            'monthly_cost_used': self.monthly_cost_used,
            'daily_requests_used': self.daily_requests_used,
            'monthly_requests_used': self.monthly_requests_used,
            'daily_cost_remaining': self.config.daily_cost_limit - self.daily_cost_used,
            'monthly_cost_remaining': self.config.monthly_cost_limit - self.monthly_cost_used,
            'daily_requests_remaining': self.config.daily_request_limit - self.daily_requests_used,
            'monthly_requests_remaining': self.config.monthly_request_limit - self.monthly_requests_used,
            'daily_cost_percent_used': (self.daily_cost_used / self.config.daily_cost_limit) * 100 if self.config.daily_cost_limit > 0 else 0,
            'monthly_cost_percent_used': (self.monthly_cost_used / self.config.monthly_cost_limit) * 100 if self.config.monthly_cost_limit > 0 else 0,
            'daily_reset_in': 'Next day',
            'monthly_reset_in': f'{days_left_in_month} days'
        }
    
    def update_config(self, new_config: QuotaConfig):
        """Update quota configuration.
        
        Args:
            new_config: New quota configuration
        """
        self.config = new_config
        logger.info(f"Quota configuration updated: {vars(new_config)}")


class PerformanceMonitor:
    """Monitor for model performance metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of requests to keep in the rolling window
        """
        self.window_size = window_size
        self.metrics = PerformanceMetrics()
        self.request_history: List[Dict[str, Any]] = []
    
    def record_request(self,
                       model_name: str,
                       success: bool,
                       latency: float,
                       tokens_used: int,
                       cost: float,
                       error_type: Optional[str] = None):
        """Record a request.
        
        Args:
            model_name: Name of the model used
            success: Whether the request was successful
            latency: Request latency in seconds
            tokens_used: Number of tokens used
            cost: Cost of the request in USD
            error_type: Type of error if request failed
        """
        # Record in metrics
        self.metrics.record_request(
            model_name=model_name,
            success=success,
            latency=latency,
            tokens_used=tokens_used,
            cost=cost,
            error_type=error_type
        )
        
        # Add to history
        request_data = {
            'timestamp': time.time(),
            'model_name': model_name,
            'success': success,
            'latency': latency,
            'tokens_used': tokens_used,
            'cost': cost,
            'error_type': error_type
        }
        
        self.request_history.append(request_data)
        
        # Trim history if needed
        if len(self.request_history) > self.window_size:
            self.request_history = self.request_history[-self.window_size:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = self.metrics.get_metrics()
        
        # Add recent trends if we have enough data
        if len(self.request_history) >= 2:
            # Calculate trends over the last 10 requests or all if less than 10
            recent_requests = self.request_history[-min(10, len(self.request_history)):]
            
            # Calculate average latency trend
            latencies = [req['latency'] for req in recent_requests]
            avg_latency = sum(latencies) / len(latencies)
            
            # Calculate success rate trend
            successes = sum(1 for req in recent_requests if req['success'])
            success_rate = successes / len(recent_requests)
            
            metrics['recent_trends'] = {
                'average_latency': avg_latency,
                'success_rate': success_rate,
                'requests_per_minute': self._calculate_requests_per_minute()
            }
        
        return metrics
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute based on recent history.
        
        Returns:
            Requests per minute
        """
        if not self.request_history:
            return 0.0
        
        # Get timestamps of all requests in the window
        timestamps = [req['timestamp'] for req in self.request_history]
        
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate time span in minutes
        time_span_minutes = (max(timestamps) - min(timestamps)) / 60.0
        
        if time_span_minutes < 0.01:  # Avoid division by very small numbers
            return 0.0
        
        return len(timestamps) / time_span_minutes


class ModelHealthChecker:
    """Checker for model health status."""
    
    def __init__(self):
        """Initialize model health checker."""
        self.model_health: Dict[str, Dict[str, Any]] = {}
    
    def record_health(self,
                      model_name: str,
                      success: bool,
                      latency: float,
                      error_type: Optional[str] = None):
        """Record health check for a model.
        
        Args:
            model_name: Name of the model
            success: Whether the health check was successful
            latency: Latency of the health check in seconds
            error_type: Type of error if health check failed
        """
        if model_name not in self.model_health:
            self.model_health[model_name] = {
                'checks': 0,
                'successful_checks': 0,
                'failed_checks': 0,
                'total_latency': 0.0,
                'last_check': None,
                'last_success': None,
                'last_failure': None,
                'last_error_type': None,
                'consecutive_failures': 0
            }
        
        health = self.model_health[model_name]
        health['checks'] += 1
        health['total_latency'] += latency
        health['last_check'] = time.time()
        
        if success:
            health['successful_checks'] += 1
            health['last_success'] = time.time()
            health['consecutive_failures'] = 0
        else:
            health['failed_checks'] += 1
            health['last_failure'] = time.time()
            health['last_error_type'] = error_type
            health['consecutive_failures'] += 1
    
    def get_model_health(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing health status or None if model not found
        """
        if model_name not in self.model_health:
            return None
        
        health = self.model_health[model_name]
        
        return {
            'success_rate': (health['successful_checks'] / health['checks']) if health['checks'] > 0 else 0,
            'average_latency': (health['total_latency'] / health['checks']) if health['checks'] > 0 else 0,
            'last_check': health['last_check'],
            'last_success': health['last_success'],
            'last_failure': health['last_failure'],
            'last_error_type': health['last_error_type'],
            'consecutive_failures': health['consecutive_failures'],
            'status': self._determine_status(health)
        }
    
    def _determine_status(self, health: Dict[str, Any]) -> str:
        """Determine the status of a model based on health metrics.
        
        Args:
            health: Health metrics for the model
            
        Returns:
            Status string: 'healthy', 'degraded', or 'unhealthy'
        """
        if health['consecutive_failures'] >= 5:
            return 'unhealthy'
        
        success_rate = (health['successful_checks'] / health['checks']) if health['checks'] > 0 else 0
        
        if success_rate < 0.5:
            return 'unhealthy'
        elif success_rate < 0.9:
            return 'degraded'
        else:
            return 'healthy'
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all models.
        
        Returns:
            Dictionary mapping model names to health status
        """
        result = {}
        
        for model_name in self.model_health:
            result[model_name] = self.get_model_health(model_name)
        
        return result