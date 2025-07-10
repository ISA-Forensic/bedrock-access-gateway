import sqlite3
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from api.setting import MODELS_DB_PATH

logger = logging.getLogger(__name__)

class ModelsDatabase:
    """SQLite database manager for models configuration"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            if MODELS_DB_PATH:
                db_path = MODELS_DB_PATH
            else:
                # Default to /app/api/data/models.db in Docker, or local path for development
                data_dir = Path("/app/api/data") if os.path.exists("/app") else Path(__file__).parent / "data"
                data_dir.mkdir(exist_ok=True)
                db_path = data_dir / "models.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"Models database initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            # Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            
            # Create models table for chat models
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    modalities TEXT,  -- JSON array as string
                    owned_by TEXT DEFAULT 'bedrock',
                    max_tokens INTEGER DEFAULT 4000,
                    temperature REAL DEFAULT 0.7,
                    top_p REAL DEFAULT 0.9,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create embedding_models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_models (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    modalities TEXT,  -- JSON array as string
                    owned_by TEXT DEFAULT 'bedrock',
                    max_tokens INTEGER DEFAULT 4000,
                    temperature REAL DEFAULT 0.7,
                    top_p REAL DEFAULT 0.9,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add max_tokens column to existing tables if it doesn't exist
            try:
                conn.execute("ALTER TABLE models ADD COLUMN max_tokens INTEGER DEFAULT 4000")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            try:
                conn.execute("ALTER TABLE embedding_models ADD COLUMN max_tokens INTEGER DEFAULT 4000")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Add temperature and top_p columns to existing tables if they don't exist
            try:
                conn.execute("ALTER TABLE models ADD COLUMN temperature REAL DEFAULT 0.7")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            try:
                conn.execute("ALTER TABLE models ADD COLUMN top_p REAL DEFAULT 0.9")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            try:
                conn.execute("ALTER TABLE embedding_models ADD COLUMN temperature REAL DEFAULT 0.7")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            try:
                conn.execute("ALTER TABLE embedding_models ADD COLUMN top_p REAL DEFAULT 0.9")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Create triggers to update updated_at automatically
            for table in ['models', 'embedding_models']:
                conn.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS update_{table}_timestamp 
                    AFTER UPDATE ON {table}
                    BEGIN
                        UPDATE {table} SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                    END
                """)
            
            conn.commit()
    
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
    
    # Chat Models CRUD
    def get_chat_models(self) -> List[Dict]:
        """Get all chat models"""
        rows = self.execute_query("SELECT * FROM models ORDER BY id")
        models = []
        for row in rows:
            model = dict(row)
            # Parse modalities JSON
            if model['modalities']:
                try:
                    model['modalities'] = json.loads(model['modalities'])
                except json.JSONDecodeError:
                    model['modalities'] = []
            else:
                model['modalities'] = []
            # Remove timestamps for API compatibility
            model.pop('created_at', None)
            model.pop('updated_at', None)
            models.append(model)
        return models
    
    def get_embedding_models(self) -> List[Dict]:
        """Get all embedding models"""
        rows = self.execute_query("SELECT * FROM embedding_models ORDER BY id")
        models = []
        for row in rows:
            model = dict(row)
            # Parse modalities JSON
            if model['modalities']:
                try:
                    model['modalities'] = json.loads(model['modalities'])
                except json.JSONDecodeError:
                    model['modalities'] = []
            else:
                model['modalities'] = []
            # Remove timestamps for API compatibility
            model.pop('created_at', None)
            model.pop('updated_at', None)
            models.append(model)
        return models
    
    def get_all_models(self) -> List[Dict]:
        """Get all models (chat + embedding)"""
        return self.get_chat_models() + self.get_embedding_models()
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID"""
        # Check in chat models first
        rows = self.execute_query("SELECT * FROM models WHERE id = ?", (model_id,))
        if rows:
            model = dict(rows[0])
            if model['modalities']:
                try:
                    model['modalities'] = json.loads(model['modalities'])
                except json.JSONDecodeError:
                    model['modalities'] = []
            model.pop('created_at', None)
            model.pop('updated_at', None)
            return model
        
        # Check in embedding models
        rows = self.execute_query("SELECT * FROM embedding_models WHERE id = ?", (model_id,))
        if rows:
            model = dict(rows[0])
            if model['modalities']:
                try:
                    model['modalities'] = json.loads(model['modalities'])
                except json.JSONDecodeError:
                    model['modalities'] = []
            model.pop('created_at', None)
            model.pop('updated_at', None)
            return model
        
        return None
    
    def add_chat_model(self, model: Dict) -> Dict:
        """Add a new chat model"""
        modalities_json = json.dumps(model.get('modalities', []))
        self.execute_update("""
            INSERT INTO models (id, name, description, modalities, owned_by, max_tokens, temperature, top_p)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model['id'],
            model.get('name'),
            model.get('description'),
            modalities_json,
            model.get('owned_by', 'bedrock'),
            model.get('max_tokens', 4000),
            model.get('temperature', 0.7),
            model.get('top_p', 0.9)
        ))
        return model
    
    def add_embedding_model(self, model: Dict) -> Dict:
        """Add a new embedding model"""
        modalities_json = json.dumps(model.get('modalities', []))
        self.execute_update("""
            INSERT INTO embedding_models (id, name, description, modalities, owned_by, max_tokens, temperature, top_p)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model['id'],
            model.get('name'),
            model.get('description'),
            modalities_json,
            model.get('owned_by', 'bedrock'),
            model.get('max_tokens', 4000),
            model.get('temperature', 0.7),
            model.get('top_p', 0.9)
        ))
        return model
    
    def update_chat_model(self, model_id: str, model: Dict) -> Dict:
        """Update an existing chat model"""
        modalities_json = json.dumps(model.get('modalities', []))
        self.execute_update("""
            UPDATE models 
            SET name = ?, description = ?, modalities = ?, owned_by = ?, max_tokens = ?, temperature = ?, top_p = ?
            WHERE id = ?
        """, (
            model.get('name'),
            model.get('description'),
            modalities_json,
            model.get('owned_by', 'bedrock'),
            model.get('max_tokens', 4000),
            model.get('temperature', 0.7),
            model.get('top_p', 0.9),
            model_id
        ))
        return {**model, 'id': model_id}
    
    def update_embedding_model(self, model_id: str, model: Dict) -> Dict:
        """Update an existing embedding model"""
        modalities_json = json.dumps(model.get('modalities', []))
        self.execute_update("""
            UPDATE embedding_models 
            SET name = ?, description = ?, modalities = ?, owned_by = ?, max_tokens = ?, temperature = ?, top_p = ?
            WHERE id = ?
        """, (
            model.get('name'),
            model.get('description'),
            modalities_json,
            model.get('owned_by', 'bedrock'),
            model.get('max_tokens', 4000),
            model.get('temperature', 0.7),
            model.get('top_p', 0.9),
            model_id
        ))
        return {**model, 'id': model_id}
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model (checks both tables)"""
        # Try chat models first
        affected = self.execute_update("DELETE FROM models WHERE id = ?", (model_id,))
        if affected > 0:
            return True
        
        # Try embedding models
        affected = self.execute_update("DELETE FROM embedding_models WHERE id = ?", (model_id,))
        return affected > 0
    
    def migrate_from_json(self, json_path: str = None):
        """Migrate data from existing JSON file to SQLite"""
        if json_path is None:
            json_path = Path(__file__).parent / "models_config.json"
        
        json_path = Path(json_path)
        if not json_path.exists():
            logger.info(f"No JSON file found at {json_path}, skipping migration")
            return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Migrate chat models
            for model in data.get('models', []):
                try:
                    # Add default max_tokens if not present
                    if 'max_tokens' not in model:
                        model['max_tokens'] = 4000
                    self.add_chat_model(model)
                    logger.info(f"Migrated chat model: {model.get('id')}")
                except Exception as e:
                    logger.warning(f"Failed to migrate chat model {model.get('id')}: {e}")
            
            # Migrate embedding models
            for model in data.get('embedding_models', []):
                try:
                    # Add default max_tokens if not present
                    if 'max_tokens' not in model:
                        model['max_tokens'] = 4000
                    self.add_embedding_model(model)
                    logger.info(f"Migrated embedding model: {model.get('id')}")
                except Exception as e:
                    logger.warning(f"Failed to migrate embedding model {model.get('id')}: {e}")
            
            logger.info(f"Migration from {json_path} completed")
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")

# Global instance
_models_db = None

def get_models_database() -> ModelsDatabase:
    """Get or create the global models database instance"""
    global _models_db
    if _models_db is None:
        _models_db = ModelsDatabase()
    return _models_db 