# Config-Based Models

This implementation allows you to define available models using a configuration file instead of dynamically fetching them from AWS Bedrock.

## Configuration File

The models are defined in `src/api/models_config.json`. You can customize this file to include only the models you want to make available.

### Structure

```json
{
  "models": [
    {
      "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
      "name": "Claude 3.5 Sonnet",
      "description": "Anthropic's most intelligent model",
      "modalities": ["TEXT", "IMAGE"],
      "owned_by": "anthropic"
    }
  ],
  "embedding_models": [
    {
      "id": "cohere.embed-multilingual-v3",
      "name": "Cohere Embed Multilingual",
      "description": "Multilingual embedding model",
      "owned_by": "cohere"
    }
  ]
}
```

### Fields

- `id`: The model identifier (required)
- `name`: Human-readable model name (optional)
- `description`: Model description (optional)
- `modalities`: List of supported modalities like "TEXT", "IMAGE" (optional)
- `owned_by`: Model provider/owner (optional)

## Environment Variables

You can customize the config file location using:

```bash
export MODELS_CONFIG_PATH="/path/to/your/models_config.json"
```

## API Endpoints

### List Models
```bash
curl -H "Authorization: Bearer bedrock" http://localhost:8000/api/v1/models
```

### Get Specific Model
```bash
curl -H "Authorization: Bearer bedrock" http://localhost:8000/api/v1/models/anthropic.claude-3-5-sonnet-20240620-v1:0
```

### Reload Configuration
```bash
curl -X POST -H "Authorization: Bearer bedrock" http://localhost:8000/api/v1/models/reload
```

## Benefits

1. **No AWS Credentials Required**: The models endpoint works without AWS credentials
2. **Customizable**: Only expose the models you want to support
3. **Fast**: No API calls to AWS Bedrock for model listing
4. **Hot Reload**: Update the config file and reload without restarting the server
5. **Rich Metadata**: Include custom names, descriptions, and other metadata

## Testing

You can test the functionality directly with Python:

```python
from api.routers.model import list_models
import asyncio
import json

# Test listing models
result = asyncio.run(list_models())
print(json.dumps(result.dict(), indent=2))
```

## Migration

This implementation maintains compatibility with the existing Bedrock integration. The chat and embeddings endpoints will still validate models against the config file and use the actual Bedrock models for inference. 