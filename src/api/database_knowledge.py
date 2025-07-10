import sqlite3
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from api.setting import KNOWLEDGE_DB_PATH

logger = logging.getLogger(__name__)

class KnowledgeDatabase:
    """SQLite database manager for knowledge base configuration"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            if KNOWLEDGE_DB_PATH:
                db_path = KNOWLEDGE_DB_PATH
            else:
                # Default to /app/api/data/knowledge.db in Docker, or local path for development
                data_dir = Path("/app/api/data") if os.path.exists("/app") else Path(__file__).parent / "data"
                data_dir.mkdir(exist_ok=True)
                db_path = data_dir / "knowledge.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Knowledge base database initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            
            # Create knowledge_bases table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    knowledge_base_id TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    num_results INTEGER DEFAULT 5,
                    search_type TEXT DEFAULT 'HYBRID',
                    max_workers INTEGER DEFAULT 3,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create settings table for default configurations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,  -- JSON value as string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create triggers to update updated_at automatically
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_knowledge_bases_timestamp 
                AFTER UPDATE ON knowledge_bases
                BEGIN
                    UPDATE knowledge_bases SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_settings_timestamp 
                AFTER UPDATE ON settings
                BEGIN
                    UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
                END
            """)
            
            # Initialize default settings if not exists
            self._init_default_settings(conn)
            
            conn.commit()
    
    def _init_default_settings(self, conn):
        """Initialize default settings"""
        default_settings = {
            "num_results": 5,
            "search_type": "HYBRID",
            "max_workers": 3,
            "system_prompt": "You are a helpful assistant. Analyze the documents provided and answer the user's question based only on the information. If the answer cannot be found, say you don't have that info."
        }
        
        # Check if default settings exist
        cursor = conn.execute("SELECT value FROM settings WHERE key = 'default_settings'")
        if not cursor.fetchone():
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('default_settings', ?)",
                (json.dumps(default_settings),)
            )
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    # Knowledge Base CRUD
    def get_knowledge_bases(self) -> List[Dict]:
        """Get all knowledge bases"""
        logger.info("=== GET KNOWLEDGE BASES FROM DATABASE ===")
        rows = self.execute_query("SELECT * FROM knowledge_bases ORDER BY id")
        knowledge_bases = []
        for row in rows:
            kb = dict(row)
            # Remove timestamps for API compatibility
            kb.pop('created_at', None)
            kb.pop('updated_at', None)
            knowledge_bases.append(kb)
        
        logger.info(f"Found {len(knowledge_bases)} knowledge bases in database")
        return knowledge_bases
    
    def get_enabled_knowledge_bases(self) -> List[Dict]:
        """Get enabled knowledge bases"""
        logger.info("=== GET ENABLED KNOWLEDGE BASES FROM DATABASE ===")
        rows = self.execute_query("SELECT * FROM knowledge_bases WHERE enabled = 1 ORDER BY id")
        knowledge_bases = []
        for row in rows:
            kb = dict(row)
            # Remove timestamps for API compatibility
            kb.pop('created_at', None)
            kb.pop('updated_at', None)
            knowledge_bases.append(kb)
        
        logger.info(f"Found {len(knowledge_bases)} enabled knowledge bases in database")
        return knowledge_bases
    
    def get_knowledge_base_by_id(self, kb_id: str) -> Optional[Dict]:
        """Get a specific knowledge base by ID"""
        rows = self.execute_query("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
        if rows:
            kb = dict(rows[0])
            # Remove timestamps for API compatibility
            kb.pop('created_at', None)
            kb.pop('updated_at', None)
            return kb
        return None
    
    def add_knowledge_base(self, kb: Dict) -> Dict:
        """Add a new knowledge base"""
        # Check if knowledge base already exists
        existing = self.get_knowledge_base_by_id(kb['id'])
        if existing:
            raise ValueError(f"Knowledge base with id '{kb['id']}' already exists")
        
        self.execute_update("""
            INSERT INTO knowledge_bases (
                id, name, description, knowledge_base_id, enabled, 
                num_results, search_type, max_workers
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kb['id'],
            kb.get('name'),
            kb.get('description'),
            kb.get('knowledge_base_id'),
            kb.get('enabled', True),
            kb.get('num_results', 5),
            kb.get('search_type', 'HYBRID'),
            kb.get('max_workers', 3)
        ))
        
        return kb
    
    def update_knowledge_base(self, kb_id: str, kb: Dict) -> Dict:
        """Update an existing knowledge base"""
        existing = self.get_knowledge_base_by_id(kb_id)
        if not existing:
            raise ValueError(f"Knowledge base '{kb_id}' not found")
        
        # Merge with existing data
        merged_kb = {**existing, **kb, 'id': kb_id}
        
        self.execute_update("""
            UPDATE knowledge_bases 
            SET name = ?, description = ?, knowledge_base_id = ?, enabled = ?,
                num_results = ?, search_type = ?, max_workers = ?
            WHERE id = ?
        """, (
            merged_kb.get('name'),
            merged_kb.get('description'),
            merged_kb.get('knowledge_base_id'),
            merged_kb.get('enabled', True),
            merged_kb.get('num_results', 5),
            merged_kb.get('search_type', 'HYBRID'),
            merged_kb.get('max_workers', 3),
            kb_id
        ))
        
        return merged_kb
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base"""
        affected = self.execute_update("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        return affected > 0
    
    def is_knowledge_base_enabled(self, kb_id: str) -> bool:
        """Check if a knowledge base is enabled"""
        kb = self.get_knowledge_base_by_id(kb_id)
        return kb.get('enabled', False) if kb else False
    
    def get_knowledge_base_ids(self) -> List[str]:
        """Get list of all knowledge base IDs"""
        rows = self.execute_query("SELECT id FROM knowledge_bases ORDER BY id")
        return [row['id'] for row in rows]
    
    def get_enabled_knowledge_base_ids(self) -> List[str]:
        """Get list of enabled knowledge base IDs"""
        rows = self.execute_query("SELECT id FROM knowledge_bases WHERE enabled = 1 ORDER BY id")
        return [row['id'] for row in rows]
    
    # Settings management
    def get_default_settings(self) -> Dict:
        """Get default settings"""
        rows = self.execute_query("SELECT value FROM settings WHERE key = 'default_settings'")
        if rows:
            try:
                return json.loads(rows[0]['value'])
            except json.JSONDecodeError:
                logger.error("Failed to parse default settings JSON")
        
        # Return default if not found or corrupted
        return {
            "num_results": 5,
            "search_type": "HYBRID",
            "max_workers": 3,
            "system_prompt": "You are a helpful assistant."
        }
    
    def update_default_settings(self, settings: Dict) -> Dict:
        """Update default settings"""
        current_settings = self.get_default_settings()
        merged_settings = {**current_settings, **settings}
        
        self.execute_update("""
            INSERT OR REPLACE INTO settings (key, value) 
            VALUES ('default_settings', ?)
        """, (json.dumps(merged_settings),))
        
        return merged_settings
    
    def migrate_from_json(self, json_path: str = None):
        """Migrate data from existing JSON file to SQLite"""
        if json_path is None:
            json_path = Path(__file__).parent / "knowledge_base_config.json"
        
        json_path = Path(json_path)
        if not json_path.exists():
            logger.info(f"No JSON file found at {json_path}, skipping migration")
            return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Migrate knowledge bases
            for kb in data.get('knowledge_bases', []):
                try:
                    # Remove max_tokens if it exists (now handled by models)
                    kb_clean = {k: v for k, v in kb.items() if k != 'max_tokens'}
                    self.add_knowledge_base(kb_clean)
                    logger.info(f"Migrated knowledge base: {kb.get('id')}")
                except Exception as e:
                    logger.warning(f"Failed to migrate knowledge base {kb.get('id')}: {e}")
            
            # Migrate default settings (remove max_tokens from defaults)
            default_settings = data.get('default_settings', {})
            if default_settings:
                # Remove max_tokens from default settings as it's now model-specific
                default_settings_clean = {k: v for k, v in default_settings.items() if k != 'max_tokens'}
                if default_settings_clean:
                    self.update_default_settings(default_settings_clean)
                    logger.info("Migrated default settings (removed max_tokens)")
            
            logger.info(f"Migration from {json_path} completed")
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")

# Global instance
_knowledge_db = None

def get_knowledge_database() -> KnowledgeDatabase:
    """Get or create the global knowledge database instance"""
    global _knowledge_db
    if _knowledge_db is None:
        _knowledge_db = KnowledgeDatabase()
    return _knowledge_db 