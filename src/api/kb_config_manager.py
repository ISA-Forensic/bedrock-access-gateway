import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from api.setting import KB_CONFIG_PATH

logger = logging.getLogger(__name__)

class KnowledgeBaseConfigManager:
    """Manages knowledge base configuration from JSON file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Use environment variable if set, otherwise default to knowledge_base_config.json in the same directory
            if KB_CONFIG_PATH:
                config_path = KB_CONFIG_PATH
            else:
                # Default to knowledge_base_config.json in the same directory as this file
                config_path = Path(__file__).parent / "knowledge_base_config.json"
        self.config_path = Path(config_path)
        
        self._config_cache = None
        self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Knowledge base config file not found at {self.config_path}")
                return {"knowledge_bases": [], "default_settings": {}}
            
            with open(self.config_path, 'r') as f:
                self._config_cache = json.load(f)
                logger.info(f"Loaded knowledge base config from {self.config_path}")
                return self._config_cache
        except Exception as e:
            logger.error(f"Error loading knowledge base config file: {e}")
            return {"knowledge_bases": [], "default_settings": {}}
    
    def get_config(self) -> Dict:
        """Get the current configuration, reloading if cache is empty"""
        if self._config_cache is None:
            self._load_config()
        return self._config_cache or {"knowledge_bases": [], "default_settings": {}}
    
    def reload_config(self) -> Dict:
        """Force reload the configuration from file"""
        self._config_cache = None
        return self._load_config()
    
    def get_knowledge_bases(self) -> List[Dict]:
        """Get list of knowledge bases"""
        config = self.get_config()
        return config.get("knowledge_bases", [])
    
    def get_enabled_knowledge_bases(self) -> List[Dict]:
        """Get list of enabled knowledge bases"""
        knowledge_bases = self.get_knowledge_bases()
        return [kb for kb in knowledge_bases if kb.get("enabled", True)]
    
    def get_knowledge_base_by_id(self, kb_id: str) -> Optional[Dict]:
        """Get a specific knowledge base by ID"""
        knowledge_bases = self.get_knowledge_bases()
        for kb in knowledge_bases:
            if kb.get("id") == kb_id:
                return kb
        return None
    
    def get_default_settings(self) -> Dict:
        """Get default settings for knowledge base operations"""
        config = self.get_config()
        return config.get("default_settings", {})
    
    def is_knowledge_base_enabled(self, kb_id: str) -> bool:
        """Check if a knowledge base is enabled"""
        kb = self.get_knowledge_base_by_id(kb_id)
        return kb.get("enabled", False) if kb else False
    
    def get_knowledge_base_ids(self) -> List[str]:
        """Get list of all knowledge base IDs"""
        knowledge_bases = self.get_knowledge_bases()
        return [kb.get("id") for kb in knowledge_bases if kb.get("id")]
    
    def get_enabled_knowledge_base_ids(self) -> List[str]:
        """Get list of enabled knowledge base IDs"""
        enabled_kbs = self.get_enabled_knowledge_bases()
        return [kb.get("id") for kb in enabled_kbs if kb.get("id")]

    def _save_config(self):
        """Persist the current cache to disk."""
        if self._config_cache is None:
            return
        try:
            with open(self.config_path, "w") as f:
                json.dump(self._config_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving knowledge base config: {e}")

    def add_knowledge_base(self, kb: Dict) -> Dict:
        """Add a new knowledge base configuration."""
        config = self.get_config()
        knowledge_bases = config.setdefault("knowledge_bases", [])
        if any(existing.get("id") == kb.get("id") for existing in knowledge_bases):
            raise ValueError(f"Knowledge base with id '{kb.get('id')}' already exists")
        knowledge_bases.append(kb)
        self._save_config()
        return kb

    def update_knowledge_base(self, kb_id: str, kb: Dict) -> Dict:
        """Update an existing knowledge base by id."""
        config = self.get_config()
        knowledge_bases = config.get("knowledge_bases", [])
        for index, existing in enumerate(knowledge_bases):
            if existing.get("id") == kb_id:
                # Merge fields
                knowledge_bases[index] = {**existing, **kb, "id": kb_id}
                self._save_config()
                return knowledge_bases[index]
        raise ValueError(f"Knowledge base '{kb_id}' not found")

    def delete_knowledge_base(self, kb_id: str):
        """Delete a knowledge base by id."""
        config = self.get_config()
        knowledge_bases = config.get("knowledge_bases", [])
        new_list = [kb for kb in knowledge_bases if kb.get("id") != kb_id]
        if len(new_list) == len(knowledge_bases):
            raise ValueError(f"Knowledge base '{kb_id}' not found")
        config["knowledge_bases"] = new_list
        self._save_config()


# Global instance - will be initialized when first imported
kb_config_manager = None

def get_kb_config_manager():
    """Get or create the global knowledge base config manager instance"""
    global kb_config_manager
    if kb_config_manager is None:
        kb_config_manager = KnowledgeBaseConfigManager()
    return kb_config_manager 