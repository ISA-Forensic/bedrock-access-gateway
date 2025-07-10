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
            description=model_config.get("description"),
            max_tokens=model_config.get("max_tokens", 4000),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9)
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
        description=model_config.get("description") if model_config else None,
        max_tokens=model_config.get("max_tokens", 4000) if model_config else 4000
    )


@router.post("/reload")
async def reload_models_config():
    """Reload the models configuration from file"""
    config_manager = get_config_manager()
    config_manager.reload_config()
    return {"message": "Models configuration reloaded successfully"}


@router.post("", response_model=Model, status_code=201)
async def create_model(model: Model):
    """Add a new model configuration"""
    config_manager = get_config_manager()
    try:
        created = config_manager.add_model(model.dict(exclude_none=True))
        return Model(**created)
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{model_id}", response_model=Model)
async def update_model(model_id: str, model: Model):
    """Update an existing model"""
    config_manager = get_config_manager()
    try:
        updated = config_manager.update_model(model_id, model.dict(exclude_none=True))
        return Model(**updated)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model configuration"""
    config_manager = get_config_manager()
    try:
        config_manager.delete_model(model_id)
        return {"status": "success"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
