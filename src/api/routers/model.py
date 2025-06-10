try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.config_manager import get_config_manager
from api.schema import Model, Models

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


async def validate_model_id(model_id: str):
    config_manager = get_config_manager()
    if not config_manager.is_model_supported(model_id):
        raise HTTPException(status_code=500, detail="Unsupported Model Id")


@router.get("", response_model=Models)
async def list_models():
    """Get list of available models from config file"""
    config_manager = get_config_manager()
    models_from_config = config_manager.get_all_models()
    model_list = [
        Model(
            id=model_config["id"],
            owned_by=model_config.get("owned_by", "bedrock"),
            name=model_config.get("name"),
            description=model_config.get("description")
        )
        for model_config in models_from_config
    ]
    return Models(data=model_list)


@router.get(
    "/{model_id}",
    response_model=Model,
)
async def get_model(
    model_id: Annotated[
        str,
        Path(description="Model ID", example="anthropic.claude-3-sonnet-20240229-v1:0"),
    ],
):
    """Get details of a specific model by ID"""
    await validate_model_id(model_id)
    config_manager = get_config_manager()
    model_config = config_manager.get_model_by_id(model_id)
    return Model(
        id=model_id,
        owned_by=model_config.get("owned_by", "bedrock") if model_config else "bedrock",
        name=model_config.get("name") if model_config else None,
        description=model_config.get("description") if model_config else None
    )


@router.post("/reload")
async def reload_models_config():
    """Reload the models configuration from file"""
    config_manager = get_config_manager()
    config_manager.reload_config()
    return {"message": "Models configuration reloaded successfully"}
