import logging
from typing import Dict, List, Optional

from api.database_knowledge import get_knowledge_database

logger = logging.getLogger(__name__)

class KnowledgeBaseConfigManager:
    """Manages knowledge base configuration using SQLite database"""
    
    def __init__(self):
        self.knowledge_db = get_knowledge_database()
        
        # Auto-migrate from JSON if database is empty
        self._auto_migrate()
    
    def _auto_migrate(self):
        """Automatically migrate from JSON if database is empty"""
        try:
            # Check if database has any knowledge bases
            if not self.knowledge_db.get_knowledge_bases():
                logger.info("Database is empty, attempting migration from JSON...")
                self.knowledge_db.migrate_from_json()
        except Exception as e:
            logger.warning(f"Auto-migration failed: {e}")
    
    def get_config(self) -> Dict:
        """Get the current configuration in JSON-compatible format"""
        return {
            "knowledge_bases": self.get_knowledge_bases(),
            "default_settings": self.get_default_settings()
        }
    
    def reload_config(self) -> Dict:
        """Force reload the configuration (for SQLite, this just returns current data)"""
        return self.get_config()
    
    def get_knowledge_bases(self) -> List[Dict]:
        """Get list of knowledge bases"""
        logger.info("=== GET KNOWLEDGE BASES ===")
        knowledge_bases = self.knowledge_db.get_knowledge_bases()
        logger.info(f"Knowledge bases from database: {knowledge_bases}")
        logger.info(f"Number of knowledge bases: {len(knowledge_bases)}")
        for i, kb in enumerate(knowledge_bases):
            logger.info(f"KB {i+1}: {kb}")
        return knowledge_bases
    
    def get_enabled_knowledge_bases(self) -> List[Dict]:
        """Get list of enabled knowledge bases"""
        return self.knowledge_db.get_enabled_knowledge_bases()
    
    def get_knowledge_base_by_id(self, kb_id: str) -> Optional[Dict]:
        """Get a specific knowledge base by ID"""
        return self.knowledge_db.get_knowledge_base_by_id(kb_id)
    
    def get_default_settings(self) -> Dict:
        """Get default settings for knowledge base operations"""
        return self.knowledge_db.get_default_settings()
    
    def is_knowledge_base_enabled(self, kb_id: str) -> bool:
        """Check if a knowledge base is enabled"""
        return self.knowledge_db.is_knowledge_base_enabled(kb_id)
    
    def get_knowledge_base_ids(self) -> List[str]:
        """Get list of all knowledge base IDs"""
        return self.knowledge_db.get_knowledge_base_ids()
    
    def get_enabled_knowledge_base_ids(self) -> List[str]:
        """Get list of enabled knowledge base IDs"""
        return self.knowledge_db.get_enabled_knowledge_base_ids()

    # CRUD operations
    def add_knowledge_base(self, kb: Dict) -> Dict:
        """Add a new knowledge base configuration."""
        return self.knowledge_db.add_knowledge_base(kb)

    def update_knowledge_base(self, kb_id: str, kb: Dict) -> Dict:
        """Update an existing knowledge base by id."""
        return self.knowledge_db.update_knowledge_base(kb_id, kb)
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base"""
        return self.knowledge_db.delete_knowledge_base(kb_id)
    
    def update_default_settings(self, settings: Dict) -> Dict:
        """Update default settings"""
        return self.knowledge_db.update_default_settings(settings)

# Global instance - will be initialized when first imported
kb_config_manager = None

def get_kb_config_manager():
    """Get or create the global knowledge base config manager instance"""
    global kb_config_manager
    if kb_config_manager is None:
        kb_config_manager = KnowledgeBaseConfigManager()
    return kb_config_manager 