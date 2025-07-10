import os

DEFAULT_API_KEYS = "bedrock"

API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = os.environ.get("DEBUG", "false").lower() != "false"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = os.environ.get("ENABLE_CROSS_REGION_INFERENCE", "true").lower() != "false"

# Database paths for SQLite databases
MODELS_DB_PATH = os.environ.get("MODELS_DB_PATH")
KNOWLEDGE_DB_PATH = os.environ.get("KNOWLEDGE_DB_PATH")

# Legacy JSON configuration paths (for migration purposes)
MODELS_CONFIG_PATH = os.environ.get("MODELS_CONFIG_PATH", "/app/api/models_config.json")
KB_CONFIG_PATH = os.environ.get("KB_CONFIG_PATH")

# Gateway configuration for chat via gateway
GATEWAY_URL = os.environ.get("GATEWAY_URL")
USE_GATEWAY = os.environ.get("USE_GATEWAY", "true").lower() == "true"

# Knowledge base configuration
USE_KNOWLEDGE_BASE = os.environ.get("USE_KNOWLEDGE_BASE", "true").lower() == "true"

UI_USERNAME = os.environ.get("UI_USERNAME", "admin")
UI_PASSWORD = os.environ.get("UI_PASSWORD", "password")
