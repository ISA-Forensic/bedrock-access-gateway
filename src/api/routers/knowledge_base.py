import logging
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Body

from api.auth import api_key_auth
from api.kb_config_manager import get_kb_config_manager
from api.schema import KnowledgeBase, KnowledgeBases

logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[Depends(api_key_auth)],
)


@router.get("/knowledge-bases", response_model=KnowledgeBases)
async def list_knowledge_bases():
    """Get list of available knowledge bases from database"""
    logger.info("=== LIST KNOWLEDGE BASES REQUEST START ===")
    try:
        logger.info("Getting knowledge base config manager...")
        kb_manager = get_kb_config_manager()
        logger.info(f"KB Manager type: {type(kb_manager)}")
        
        logger.info("Fetching knowledge bases configuration...")
        knowledge_bases_config = kb_manager.get_knowledge_bases()
        logger.info(f"Raw knowledge bases config: {knowledge_bases_config}")
        logger.info(f"Number of knowledge bases found: {len(knowledge_bases_config)}")
        
        kb_list = []
        for i, kb_config in enumerate(knowledge_bases_config):
            logger.info(f"Processing KB {i+1}: {kb_config}")
            try:
                kb_obj = KnowledgeBase(
                    id=kb_config["id"],
                    name=kb_config.get("name"),
                    description=kb_config.get("description"),
                    knowledge_base_id=kb_config.get("knowledge_base_id"),
                    enabled=kb_config.get("enabled", True),
                    num_results=kb_config.get("num_results", 5),
                    search_type=kb_config.get("search_type", "HYBRID")
                )
                logger.info(f"Created KnowledgeBase object {i+1}: {kb_obj.model_dump()}")
                kb_list.append(kb_obj)
            except Exception as kb_error:
                logger.error(f"Error creating KnowledgeBase object {i+1}: {kb_error}")
                logger.error(f"KB config that failed: {kb_config}")
                import traceback
                logger.error(f"KB creation traceback: {traceback.format_exc()}")
                raise
        
        logger.info(f"Successfully created {len(kb_list)} KnowledgeBase objects")
        result = KnowledgeBases(data=kb_list)
        logger.info(f"Final result: {result.model_dump()}")
        logger.info("=== LIST KNOWLEDGE BASES REQUEST SUCCESS ===")
        return result
    except Exception as e:
        logger.error(f"=== LIST KNOWLEDGE BASES REQUEST FAILED ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch knowledge bases: {str(e)}")


@router.get("/knowledge-bases/enabled", response_model=KnowledgeBases)
async def list_enabled_knowledge_bases():
    """Get list of enabled knowledge bases from database"""
    logger.info("=== LIST ENABLED KNOWLEDGE BASES REQUEST START ===")
    try:
        logger.info("Getting knowledge base config manager...")
        kb_manager = get_kb_config_manager()
        
        logger.info("Fetching enabled knowledge bases configuration...")
        enabled_kbs_config = kb_manager.get_enabled_knowledge_bases()
        logger.info(f"Enabled knowledge bases config: {enabled_kbs_config}")
        logger.info(f"Number of enabled knowledge bases: {len(enabled_kbs_config)}")
        
        kb_list = []
        for i, kb_config in enumerate(enabled_kbs_config):
            logger.info(f"Processing enabled KB {i+1}: {kb_config}")
            kb_obj = KnowledgeBase(
                id=kb_config["id"],
                name=kb_config.get("name"),
                description=kb_config.get("description"),
                knowledge_base_id=kb_config.get("knowledge_base_id"),
                enabled=kb_config.get("enabled", True),
                num_results=kb_config.get("num_results", 5),
                search_type=kb_config.get("search_type", "HYBRID")
            )
            logger.info(f"Created enabled KnowledgeBase object {i+1}: {kb_obj.model_dump()}")
            kb_list.append(kb_obj)
        
        result = KnowledgeBases(data=kb_list)
        logger.info(f"Enabled KBs result: {result.model_dump()}")
        logger.info("=== LIST ENABLED KNOWLEDGE BASES REQUEST SUCCESS ===")
        return result
    except Exception as e:
        logger.error(f"=== LIST ENABLED KNOWLEDGE BASES REQUEST FAILED ===")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    """Get default knowledge base settings"""
    try:
        kb_manager = get_kb_config_manager()
        return kb_manager.get_default_settings()
    except Exception as e:
        logger.error(f"Error getting default settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-bases/reload")
async def reload_knowledge_base_config() -> Dict[str, str]:
    """Reload the knowledge base configuration from database"""
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


@router.post("/knowledge-bases", response_model=KnowledgeBase, status_code=201)
async def create_knowledge_base(kb: KnowledgeBase):
    """Create a new knowledge base entry"""
    kb_manager = get_kb_config_manager()
    try:
        created = kb_manager.add_knowledge_base(kb.dict(exclude_none=True))
        return KnowledgeBase(**created)
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-bases/{kb_id}", response_model=KnowledgeBase)
async def update_knowledge_base(kb_id: str, kb: KnowledgeBase):
    """Update an existing knowledge base"""
    kb_manager = get_kb_config_manager()
    try:
        updated = kb_manager.update_knowledge_base(kb_id, kb.dict(exclude_none=True))
        return KnowledgeBase(**updated)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating knowledge base {kb_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(kb_id: str) -> Dict[str, str]:
    """Delete a knowledge base"""
    kb_manager = get_kb_config_manager()
    try:
        success = kb_manager.delete_knowledge_base(kb_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
        return {"message": f"Knowledge base '{kb_id}' deleted successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error deleting knowledge base {kb_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-bases/settings/default")
async def update_default_settings(settings: Dict = Body(...)):
    """Update default knowledge base settings"""
    kb_manager = get_kb_config_manager()
    try:
        updated_settings = kb_manager.update_default_settings(settings)
        return updated_settings
    except Exception as e:
        logger.error(f"Error updating default settings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 