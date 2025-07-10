"""
Monitoring and metrics collection for Bedrock Access Gateway
Tracks costs, token usage, API performance, and errors
"""
import time
import structlog
from typing import Dict, Optional, Any
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from functools import wraps
import json
import os
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=False) if os.getenv("DEBUG", "false").lower() == "true" 
        else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("bedrock_gateway")

# Prometheus Metrics
# ===================

# API Request Metrics
api_requests_total = Counter(
    'bedrock_api_requests_total',
    'Total API requests to Bedrock Gateway',
    ['method', 'endpoint', 'status_code', 'model']
)

api_request_duration = Histogram(
    'bedrock_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint', 'model']
)

# Bedrock-specific Metrics
bedrock_requests_total = Counter(
    'bedrock_requests_total',
    'Total requests to AWS Bedrock',
    ['model', 'operation', 'region', 'status']
)

bedrock_request_duration = Histogram(
    'bedrock_request_duration_seconds',
    'AWS Bedrock request duration in seconds',
    ['model', 'operation']
)

# Token Usage Metrics
tokens_used_total = Counter(
    'bedrock_tokens_used_total',
    'Total tokens used',
    ['model', 'token_type', 'user', 'project']  # token_type: input, output, total
)

# Cost Metrics (estimated)
estimated_costs_total = Counter(
    'bedrock_estimated_costs_total',
    'Estimated costs in USD',
    ['model', 'cost_type', 'user', 'project']  # cost_type: input, output, total
)

# Knowledge Base Metrics
kb_requests_total = Counter(
    'bedrock_kb_requests_total',
    'Total knowledge base requests',
    ['knowledge_base_id', 'search_type', 'status']
)

kb_request_duration = Histogram(
    'bedrock_kb_request_duration_seconds',
    'Knowledge base request duration',
    ['knowledge_base_id', 'search_type']
)

kb_documents_retrieved = Histogram(
    'bedrock_kb_documents_retrieved',
    'Number of documents retrieved from knowledge base',
    ['knowledge_base_id', 'search_type']
)

# Error Metrics
errors_total = Counter(
    'bedrock_errors_total',
    'Total errors',
    ['error_type', 'model', 'endpoint']
)

# Active Connections
active_connections = Gauge(
    'bedrock_active_connections',
    'Number of active connections'
)

# Model Usage
model_usage = Counter(
    'bedrock_model_usage_total',
    'Model usage count',
    ['model', 'user', 'project']
)

# System Info
system_info = Info(
    'bedrock_gateway_info',
    'System information'
)

# Pricing Information (approximate as of 2024)
MODEL_PRICING = {
    # Claude models (per 1K tokens)
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-instant-v1": {"input": 0.0008, "output": 0.0024},
    
    # Titan models
    "amazon.titan-text-express-v1": {"input": 0.0008, "output": 0.0016},
    "amazon.titan-text-lite-v1": {"input": 0.0003, "output": 0.0004},
    
    # Cohere models
    "cohere.command-text-v14": {"input": 0.0015, "output": 0.002},
    "cohere.command-light-text-v14": {"input": 0.0003, "output": 0.0006},
    
    # Embedding models (per 1K tokens)
    "cohere.embed-multilingual-v3": {"input": 0.0001, "output": 0.0},
    "cohere.embed-english-v3": {"input": 0.0001, "output": 0.0},
    "amazon.titan-embed-text-v2:0": {"input": 0.00002, "output": 0.0},
}

class MetricsCollector:
    """Central metrics collection class"""
    
    def __init__(self):
        self.logger = logger
        self._start_time = time.time()
        
        # Set system info
        system_info.info({
            'version': '1.0.0',
            'python_version': os.sys.version,
            'start_time': datetime.now().isoformat()
        })
    
    def calculate_cost(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> Dict[str, float]:
        """Calculate estimated cost for token usage"""
        pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})  # Default pricing
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, model: str = "unknown"):
        """Record API request metrics"""
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            model=model
        ).inc()
        
        api_request_duration.labels(
            method=method,
            endpoint=endpoint,
            model=model
        ).observe(duration)
        
        self.logger.info(
            "API request completed",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration,
            model=model
        )
    
    def record_bedrock_request(self, model: str, operation: str, region: str,
                              status: str, duration: float, 
                              input_tokens: int = 0, output_tokens: int = 0,
                              user: str = "unknown", project: str = "default"):
        """Record Bedrock API usage and costs"""
        
        # Record request metrics
        bedrock_requests_total.labels(
            model=model,
            operation=operation,
            region=region,
            status=status
        ).inc()
        
        bedrock_request_duration.labels(
            model=model,
            operation=operation
        ).observe(duration)
        
        # Record token usage
        if input_tokens > 0:
            tokens_used_total.labels(
                model=model,
                token_type="input",
                user=user,
                project=project
            ).inc(input_tokens)
        
        if output_tokens > 0:
            tokens_used_total.labels(
                model=model,
                token_type="output",
                user=user,
                project=project
            ).inc(output_tokens)
        
        total_tokens = input_tokens + output_tokens
        if total_tokens > 0:
            tokens_used_total.labels(
                model=model,
                token_type="total",
                user=user,
                project=project
            ).inc(total_tokens)
        
        # Calculate and record costs
        costs = self.calculate_cost(model, input_tokens, output_tokens)
        
        if costs["input_cost"] > 0:
            estimated_costs_total.labels(
                model=model,
                cost_type="input",
                user=user,
                project=project
            ).inc(costs["input_cost"])
        
        if costs["output_cost"] > 0:
            estimated_costs_total.labels(
                model=model,
                cost_type="output",
                user=user,
                project=project
            ).inc(costs["output_cost"])
        
        if costs["total_cost"] > 0:
            estimated_costs_total.labels(
                model=model,
                cost_type="total",
                user=user,
                project=project
            ).inc(costs["total_cost"])
        
        # Record model usage
        model_usage.labels(
            model=model,
            user=user,
            project=project
        ).inc()
        
        # Log detailed information
        self.logger.info(
            "Bedrock request completed",
            model=model,
            operation=operation,
            region=region,
            status=status,
            duration=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=costs["total_cost"],
            user=user,
            project=project
        )
    
    def record_knowledge_base_request(self, knowledge_base_id: str, search_type: str,
                                    status: str, duration: float, documents_count: int = 0):
        """Record knowledge base usage"""
        
        kb_requests_total.labels(
            knowledge_base_id=knowledge_base_id,
            search_type=search_type,
            status=status
        ).inc()
        
        kb_request_duration.labels(
            knowledge_base_id=knowledge_base_id,
            search_type=search_type
        ).observe(duration)
        
        if documents_count > 0:
            kb_documents_retrieved.labels(
                knowledge_base_id=knowledge_base_id,
                search_type=search_type
            ).observe(documents_count)
        
        self.logger.info(
            "Knowledge base request completed",
            knowledge_base_id=knowledge_base_id,
            search_type=search_type,
            status=status,
            duration=duration,
            documents_retrieved=documents_count
        )
    
    def record_error(self, error_type: str, model: str = "unknown", 
                    endpoint: str = "unknown", details: str = ""):
        """Record error metrics"""
        errors_total.labels(
            error_type=error_type,
            model=model,
            endpoint=endpoint
        ).inc()
        
        self.logger.error(
            "Error recorded",
            error_type=error_type,
            model=model,
            endpoint=endpoint,
            details=details
        )
    
    def increment_active_connections(self):
        """Increment active connections"""
        active_connections.inc()
    
    def decrement_active_connections(self):
        """Decrement active connections"""
        active_connections.dec()

# Global metrics collector instance
metrics = MetricsCollector()

def monitor_api_call(endpoint_name: str = ""):
    """Decorator to monitor API calls"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            model = "unknown"
            
            try:
                # Try to extract model from request if available
                if hasattr(args[0], 'model'):
                    model = args[0].model
                elif 'chat_request' in kwargs and hasattr(kwargs['chat_request'], 'model'):
                    model = kwargs['chat_request'].model
                
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                metrics.record_error(
                    error_type=type(e).__name__,
                    model=model,
                    endpoint=endpoint_name,
                    details=str(e)
                )
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_api_request(
                    method="POST",  # Most API calls are POST
                    endpoint=endpoint_name,
                    status_code=status_code,
                    duration=duration,
                    model=model
                )
        
        return wrapper
    return decorator

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return False 