import logging
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.kb_config_manager import get_kb_config_manager
from api.schema import KnowledgeBase, KnowledgeBases

logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[Depends(api_key_auth)],
)


@router.get("/knowledge-bases", response_model=KnowledgeBases)
async def list_knowledge_bases():
    """Get list of available knowledge bases from config file"""
    try:
        kb_manager = get_kb_config_manager()
        knowledge_bases_config = kb_manager.get_knowledge_bases()
        
        kb_list = [
            KnowledgeBase(
                id=kb_config["id"],
                name=kb_config.get("name"),
                description=kb_config.get("description"),
                knowledge_base_id=kb_config.get("knowledge_base_id"),
                enabled=kb_config.get("enabled", True),
                num_results=kb_config.get("num_results", 5),
                search_type=kb_config.get("search_type", "HYBRID")
            )
            for kb_config in knowledge_bases_config
        ]
        return KnowledgeBases(data=kb_list)
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/enabled", response_model=KnowledgeBases)
async def list_enabled_knowledge_bases():
    """Get list of enabled knowledge bases from config file"""
    try:
        kb_manager = get_kb_config_manager()
        enabled_kbs_config = kb_manager.get_enabled_knowledge_bases()
        
        kb_list = [
            KnowledgeBase(
                id=kb_config["id"],
                name=kb_config.get("name"),
                description=kb_config.get("description"),
                knowledge_base_id=kb_config.get("knowledge_base_id"),
                enabled=kb_config.get("enabled", True),
                num_results=kb_config.get("num_results", 5),
                search_type=kb_config.get("search_type", "HYBRID")
            )
            for kb_config in enabled_kbs_config
        ]
        return KnowledgeBases(data=kb_list)
    except Exception as e:
        logger.error(f"Error listing enabled knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBase)
async def get_knowledge_base(
    kb_id: Annotated[
        str,
        Path(description="Knowledge Base ID", example="default-kb"),
    ],
):
    """Get details of a specific knowledge base by ID"""
    try:
        kb_manager = get_kb_config_manager()
        kb_config = kb_manager.get_knowledge_base_by_id(kb_id)
        
        if not kb_config:
            raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
        
        return KnowledgeBase(
            id=kb_config["id"],
            name=kb_config.get("name"),
            description=kb_config.get("description"),
            knowledge_base_id=kb_config.get("knowledge_base_id"),
            enabled=kb_config.get("enabled", True),
            num_results=kb_config.get("num_results", 5),
            search_type=kb_config.get("search_type", "HYBRID")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/settings/default")
async def get_default_settings() -> Dict:
    """Get default settings for knowledge base operations"""
    try:
        kb_manager = get_kb_config_manager()
        default_settings = kb_manager.get_default_settings()
        
        return default_settings
    except Exception as e:
        logger.error(f"Error getting default settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-bases/reload")
async def reload_knowledge_base_config() -> Dict[str, str]:
    """Reload the knowledge base configuration from file"""
    try:
        kb_manager = get_kb_config_manager()
        config = kb_manager.reload_config()
        
        num_kbs = len(config.get("knowledge_bases", []))
        return {
            "message": f"Knowledge base configuration reloaded successfully. Found {num_kbs} knowledge bases.",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error reloading knowledge base config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 