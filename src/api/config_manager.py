import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from api.setting import MODELS_CONFIG_PATH

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages model configuration from JSON file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Use environment variable if set, otherwise default to models_config.json in the same directory
            if MODELS_CONFIG_PATH:
                config_path = MODELS_CONFIG_PATH
            else:
                # Default to models_config.json in the same directory as this file
                config_path = Path(__file__).parent / "models_config.json"
        self.config_path = Path(config_path)
        
        self._config_cache = None
        self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found at {self.config_path}")
                return {"models": [], "embedding_models": []}
            
            with open(self.config_path, 'r') as f:
                self._config_cache = json.load(f)
                logger.info(f"Loaded model config from {self.config_path}")
                return self._config_cache
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {"models": [], "embedding_models": []}
    
    def get_config(self) -> Dict:
        """Get the current configuration, reloading if cache is empty"""
        if self._config_cache is None:
            self._load_config()
        return self._config_cache or {"models": [], "embedding_models": []}
    
    def reload_config(self) -> Dict:
        """Force reload the configuration from file"""
        self._config_cache = None
        return self._load_config()
    
    def get_chat_models(self) -> List[Dict]:
        """Get list of chat models"""
        config = self.get_config()
        return config.get("models", [])
    
    def get_embedding_models(self) -> List[Dict]:
        """Get list of embedding models"""
        config = self.get_config()
        return config.get("embedding_models", [])
    
    def get_all_models(self) -> List[Dict]:
        """Get all models (chat + embedding)"""
        chat_models = self.get_chat_models()
        embedding_models = self.get_embedding_models()
        return chat_models + embedding_models
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID"""
        all_models = self.get_all_models()
        for model in all_models:
            if model.get("id") == model_id:
                return model
        return None
    
    def is_model_supported(self, model_id: str) -> bool:
        """Check if a model ID is supported"""
        return self.get_model_by_id(model_id) is not None
    
    def get_model_ids(self) -> List[str]:
        """Get list of all supported model IDs"""
        all_models = self.get_all_models()
        return [model.get("id") for model in all_models if model.get("id")]

    def _save_config(self):
        """Persist the current cache to disk."""
        if self._config_cache is None:
            return
        try:
            with open(self.config_path, "w") as f:
                json.dump(self._config_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model config: {e}")

    # ----- CRUD for models -----
    def add_model(self, model: Dict) -> Dict:
        """Add a chat or embedding model."""
        config = self.get_config()
        all_models = self.get_all_models()
        if any(m.get("id") == model.get("id") for m in all_models):
            raise ValueError(f"Model with id '{model.get('id')}' already exists")
        # Decide list based on whether id contains ':embedding' maybe? Simpler: use models list by default.
        config.setdefault("models", []).append(model)
        self._save_config()
        return model

    def update_model(self, model_id: str, model: Dict) -> Dict:
        config = self.get_config()
        for key in ("models", "embedding_models"):
            models_list = config.get(key, [])
            for idx, existing in enumerate(models_list):
                if existing.get("id") == model_id:
                    models_list[idx] = {**existing, **model, "id": model_id}
                    self._save_config()
                    return models_list[idx]
        raise ValueError(f"Model '{model_id}' not found")

    def delete_model(self, model_id: str):
        config = self.get_config()
        changed = False
        for key in ("models", "embedding_models"):
            models_list = config.get(key, [])
            new_list = [m for m in models_list if m.get("id") != model_id]
            if len(new_list) != len(models_list):
                config[key] = new_list
                changed = True
        if not changed:
            raise ValueError(f"Model '{model_id}' not found")
        self._save_config()


# Global instance - will be initialized when first imported
config_manager = None

def get_config_manager():
    """Get or create the global config manager instance"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager 