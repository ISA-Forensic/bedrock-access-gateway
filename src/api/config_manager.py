import logging
from typing import Dict, List, Optional

from api.database_models import get_models_database

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages model configuration using SQLite database"""
    
    def __init__(self):
        self.models_db = get_models_database()
        
        # Auto-migrate from JSON if database is empty
        self._auto_migrate()
    
    def _auto_migrate(self):
        """Automatically migrate from JSON if database is empty"""
        try:
            # Check if database has any models
            if not self.models_db.get_all_models():
                logger.info("Database is empty, attempting migration from JSON...")
                self.models_db.migrate_from_json()
        except Exception as e:
            logger.warning(f"Auto-migration failed: {e}")
    
    def get_config(self) -> Dict:
        """Get the current configuration in JSON-compatible format"""
        return {
            "models": self.get_chat_models(),
            "embedding_models": self.get_embedding_models()
        }
    
    def reload_config(self) -> Dict:
        """Force reload the configuration (for SQLite, this just returns current data)"""
        return self.get_config()
    
    def get_chat_models(self) -> List[Dict]:
        """Get list of chat models"""
        return self.models_db.get_chat_models()
    
    def get_embedding_models(self) -> List[Dict]:
        """Get list of embedding models"""
        return self.models_db.get_embedding_models()
    
    def get_all_models(self) -> List[Dict]:
        """Get all models (chat + embedding)"""
        return self.models_db.get_all_models()
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID"""
        return self.models_db.get_model_by_id(model_id)
    
    def is_model_supported(self, model_id: str) -> bool:
        """Check if a model ID is supported"""
        return self.get_model_by_id(model_id) is not None
    
    def get_model_ids(self) -> List[str]:
        """Get list of all supported model IDs"""
        all_models = self.get_all_models()
        return [model.get("id") for model in all_models if model.get("id")]

    # CRUD operations
    def add_model(self, model: Dict) -> Dict:
        """Add a new model (auto-detects if it's chat or embedding)"""
        # For now, assume it's a chat model unless specified
        # You could add logic to detect based on model ID or other criteria
        return self.models_db.add_chat_model(model)
    
    def add_chat_model(self, model: Dict) -> Dict:
        """Add a new chat model"""
        return self.models_db.add_chat_model(model)
    
    def add_embedding_model(self, model: Dict) -> Dict:
        """Add a new embedding model"""
        return self.models_db.add_embedding_model(model)

    def update_model(self, model_id: str, model: Dict) -> Dict:
        """Update an existing model"""
        # Try to update as chat model first
        try:
            return self.models_db.update_chat_model(model_id, model)
        except Exception:
            # If that fails, try as embedding model
            return self.models_db.update_embedding_model(model_id, model)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        return self.models_db.delete_model(model_id)

# Global instance - will be initialized when first imported
config_manager = None

def get_config_manager():
    """Get or create the global config manager instance"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager 