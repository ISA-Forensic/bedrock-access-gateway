import logging
import os
import time
import concurrent.futures
try:
    from typing import Annotated, Any, Dict, List, Optional, Tuple
except ImportError:
    from typing_extensions import Annotated
    from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL, GATEWAY_URL, USE_GATEWAY, USE_KNOWLEDGE_BASE
from api.kb_config_manager import get_kb_config_manager
from api.retrieve_config import get_retrieve_config

logger = logging.getLogger(__name__)


class ChatMessage:
    """
    Simple chat message class to store role and content.
    role -> 'system', 'user', or 'assistant'
    content -> text from user or assistant
    """
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def convert_messages_to_bedrock_format(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert ChatMessage objects to Bedrock's recognized format:
    [
      { "role": "user", "content": [ {"text": "Hello!"} ] },
      ...
    ]
    """
    bedrock_messages = []
    for msg in messages:
        bedrock_messages.append({
            "role": msg.role,
            "content": [{"text": msg.content}]
        })
    return bedrock_messages


def chat_via_gateway(
    query: str,
    gateway_url: str,
    user_name: str = "unknown_user",
    project_name: str = "unknown_project",
    system_prompt: str = None,
    conversation_history: Optional[List[ChatMessage]] = None,
    inference_config: Dict[str, Any] = None,
    model_id: str = "amazon.nova-pro-v1:0",
) -> Dict[str, Any]:
    """
    Call your API Gateway + Lambda to run a direct chat with a Bedrock model.

    Args:
        query: The user query.
        gateway_url: The API Gateway endpoint.
        user_name: Tag for usage tracking.
        project_name: Tag for usage tracking.
        system_prompt: System instructions for the model.
        conversation_history: List of previous messages (ChatMessage objects).
        inference_config: Model hyperparameters (maxTokens, temperature, etc.).
        model_id: Which Bedrock model to use.

    Returns:
        Dict with fields:
          - response: Assistant text
          - input_tokens
          - output_tokens
          - elapsed_time
    """
    start_time = time.time()

    if conversation_history is None:
        conversation_history = []
    if inference_config is None:
        inference_config = {
            "maxTokens": 8192,
            "temperature": 0.7,
            "topP": 0.9
        }

    # Step 1: Build the final list of messages
    # Optionally include a "system" prompt (the "system" role in ChatMessage).
    bedrock_system = []
    if system_prompt:
        bedrock_system = [{"text": system_prompt}]

    # Convert conversation history to the required format
    messages = convert_messages_to_bedrock_format(conversation_history)

    # Append the user's new query
    messages.append({
        "role": "user",
        "content": [{"text": query}]
    })

    # Step 2: Send request to your API Gateway
    payload = {
        "action": "converse",
        "payload": {
            "modelId": model_id,
            "messages": messages,
            "system": bedrock_system,
            "inferenceConfig": inference_config,
            "user": user_name,
            "project": project_name
        }
    }

    try:
        api_key = os.getenv('BEDROCK_API_KEY')
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        resp = requests.post(gateway_url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ValueError(
                f"Lambda responded with {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        assistant_text = data.get("assistantText", "")
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("inputTokens", 0)
        output_tokens = usage_data.get("outputTokens", 0)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Chat success! input_tokens={input_tokens}, "
            f"output_tokens={output_tokens}, time={elapsed_time:.2f}s"
        )

        return {
            "response": assistant_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_time": elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during chat: {str(e)}")
        return {
            "response": f"An error occurred: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_time": elapsed_time
        }


def format_retrieved_documents(source_texts_pairs: List[Tuple[str, List[str]]], sources_metadata: List[str] = None) -> str:
    """
    Format retrieved documents into a structured XML format.
    """
    combined_context = "<documents>\n"
    chunk_index = 0
    
    for doc_index, (source, texts) in enumerate(source_texts_pairs, start=1):
        if not texts:
            logger.warning(f"No texts found for source: {source}")
            continue
            
        for i, chunk in enumerate(texts, start=1):
            # Use specific source from metadata if available
            specific_source = sources_metadata[chunk_index] if sources_metadata and chunk_index < len(sources_metadata) else source
            # Extract filename from S3 path for cleaner display
            source_filename = specific_source.split('/')[-1] if '/' in specific_source else specific_source
            
            combined_context += f"""  <document index="{chunk_index + 1}">
    <source>{source_filename}</source>
    <full_source_path>{specific_source}</full_source_path>
    <content>{chunk}</content>
  </document>
"""
            chunk_index += 1
    
    combined_context += "</documents>"
    return combined_context


def chat_via_gateway_with_kb(
    query: str,
    knowledge_base_id: str,
    gateway_url: str,
    user_name: str = "unknown_user",
    project_name: str = "unknown_project",
    system_prompt: str = None,
    conversation_history: Optional[List[ChatMessage]] = None,
    inference_config: Dict[str, Any] = None,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    num_results: int = 5,
    search_type: str = "HYBRID"
) -> Dict[str, Any]:
    """
    Main function to retrieve info from a knowledge base AND generate a response using Bedrock
    (through our Gateway API & Lambda).
    """
    start_time = time.time()
    
    if conversation_history is None:
        conversation_history = []

    if inference_config is None:
        inference_config = {
            "maxTokens": 4096,
            "temperature": 0.7,
            "topP": 0.9
        }

    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Analyze the documents provided "
            "and answer the user's question based only on the information in the documents. "
            "Always cite your sources by mentioning the document names when referencing information. "
            "If the answer cannot be found in the documents, say you don't have that information."
        )

    # 1) Retrieve information from knowledge base
    all_responses = []
    source_names = []

    logger.info(f"Starting retrieval from knowledge base")
    
    try:
        source, texts, metadata = get_retrieve_config(
            query,
            knowledge_base_id,
            gateway_url,
            user_name,
            project_name,
            num_results,
            search_type
        )
        if texts:
            all_responses.append((source, texts))
            source_names.append(source)
            logger.info(f"Retrieved {len(texts)} chunks from source: {source}")
    except Exception as e:
        logger.error(f"Error retrieving info from knowledge base: {str(e)}")

    if not all_responses:
        logger.warning("No relevant information found for the query.")
        return {
            "response": "I couldn't find any relevant information.",
            "source_documents": [],
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_time": time.time() - start_time
        }

    # 2) Format retrieved documents
    sources_from_metadata = metadata.get("sources", []) if 'metadata' in locals() else []
    documents_xml = format_retrieved_documents(all_responses, sources_from_metadata)
    sources_list = ", ".join(source_names)

    # 3) Prepare final prompt
    formatted_prompt = f"""
<user_question>
{query}
</user_question>
{documents_xml}

Please analyze the documents and answer the question based on the information provided. 
When referencing information, please include the source document name at the end of your response.
Make your answer comprehensive and cite the relevant sources.
"""

    # 4) Construct messages
    system_message = [{"text": system_prompt}]

    # Convert conversation history
    messages = convert_messages_to_bedrock_format(conversation_history)

    # Append the user query with retrieved info as the last message
    messages.append({
        "role": "user",
        "content": [{"text": formatted_prompt}]
    })

    # 5) Call the Gateway API for "converse"
    payload = {
        "action": "converse",
        "payload": {
            "modelId": model_id,
            "messages": messages,
            "system": system_message,
            "inferenceConfig": inference_config,
            "user": user_name,
            "project": project_name
        }
    }

    try:
        api_key = os.getenv('BEDROCK_API_KEY')
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        resp = requests.post(gateway_url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ValueError(f"Lambda responded with {resp.status_code}: {resp.text}")

        data = resp.json()
        assistant_text = data.get("assistantText", "")
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("inputTokens", 0)
        output_tokens = usage_data.get("outputTokens", 0)

        elapsed_time = time.time() - start_time
        logger.info(f"Converse call success! InputTokens={input_tokens}, OutputTokens={output_tokens}")

        # Build final result with individual source information
        source_documents = []
        sources_from_metadata = metadata.get("sources", [])
        
        for source, texts in all_responses:
            for i, text in enumerate(texts):
                # Use the specific source path if available, otherwise use generic source name
                specific_source = sources_from_metadata[i] if i < len(sources_from_metadata) else source
                source_documents.append({
                    "source": specific_source,
                    "text": text
                })

        return {
            "response": assistant_text,
            "source_documents": source_documents,
            "sources": source_names,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_time": elapsed_time
        }

    except Exception as e:
        logger.error(f"Error during model invocation: {str(e)}")
        return {
            "response": f"An error occurred: {str(e)}",
            "source_documents": [],
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_time": time.time() - start_time
        }


def convert_openai_messages_to_chat_messages(openai_messages) -> Tuple[List[ChatMessage], str]:
    """
    Convert OpenAI format messages to ChatMessage objects.
    Returns (conversation_history, system_prompt)
    """
    conversation_history = []
    system_prompt = None
    
    for msg in openai_messages:
        role = msg.role
        # Handle content - it can be string or list
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list) and len(msg.content) > 0:
            # Extract text from first text content item
            for item in msg.content:
                if hasattr(item, 'type') and item.type == 'text':
                    content = item.text
                    break
            else:
                content = str(msg.content)  # Fallback
        else:
            content = str(msg.content)
        
        if role == "system":
            system_prompt = content
        else:
            conversation_history.append(ChatMessage(role, content))
    
    return conversation_history, system_prompt


def convert_chat_result_to_openai_response(
    chat_result: Dict[str, Any], 
    model: str, 
    message_id: str
) -> ChatResponse:
    """
    Convert the gateway chat result to OpenAI format response.
    """
    from api.schema import ChatResponse, ChatResponseMessage, Choice, Usage
    
    # Create the response message
    response_message = ChatResponseMessage(
        role="assistant",
        content=chat_result["response"]
    )
    
    # Create the choice
    choice = Choice(
        index=0,
        message=response_message,
        finish_reason="stop"
    )
    
    # Create usage info
    usage = Usage(
        prompt_tokens=chat_result["input_tokens"],
        completion_tokens=chat_result["output_tokens"],
        total_tokens=chat_result["input_tokens"] + chat_result["output_tokens"]
    )
    
    # Create the full response
    return ChatResponse(
        id=message_id,
        model=model,
        choices=[choice],
        usage=usage
    )


router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post(
    "/completions", response_model_exclude_unset=True
)
async def chat_completions(
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "knowledge_base_id": "PLD6ZMTYSU",
                    "user_name": "John Doe"
                }
            ],
        ),
    ],
):
    # Log all incoming requests to the completions endpoint
    logger.info(f"=== CHAT COMPLETIONS REQUEST ===")
    logger.info(f"Model: {chat_request.model}")
    logger.info(f"Knowledge Base ID: {chat_request.knowledge_base_id}")
    logger.info(f"User Name: {chat_request.user_name}")
    logger.info(f"Stream: {chat_request.stream}")
    logger.info(f"Temperature: {chat_request.temperature}")
    logger.info(f"Max Tokens: {chat_request.max_tokens}")
    logger.info(f"Top P: {chat_request.top_p}")
    logger.info(f"Messages: {len(chat_request.messages)} messages")
    for i, msg in enumerate(chat_request.messages):
        content_str = str(msg.content)
        content_preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
        logger.info(f"  [{i+1}] {msg.role}: {content_preview}")
    logger.info(f"=== END REQUEST ===")
    
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    
    # Check if we should use the gateway pattern
    if USE_GATEWAY and GATEWAY_URL:
        return await chat_completions_via_gateway(chat_request)
    
    # Use the original Bedrock model approach
    if chat_request.stream:
        return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
    return await model.chat(chat_request)


async def chat_completions_via_gateway(chat_request: ChatRequest) -> ChatResponse:
    """
    Handle chat completions using the gateway pattern.
    """
    # Log the incoming request for debugging
    logger.info(f"Chat completions request received:")
    logger.info(f"  Model: {chat_request.model}")
    logger.info(f"  Knowledge Base ID: {chat_request.knowledge_base_id}")
    logger.info(f"  User Name: {chat_request.user_name}")
    logger.info(f"  Temperature: {chat_request.temperature}")
    logger.info(f"  Max Tokens: {chat_request.max_tokens}")
    logger.info(f"  Messages count: {len(chat_request.messages)}")
    for i, msg in enumerate(chat_request.messages):
        content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
        logger.info(f"    Message {i+1}: {msg.role} - {content_preview}")
    
    try:
        # Convert OpenAI messages to ChatMessage objects
        conversation_history, system_prompt = convert_openai_messages_to_chat_messages(chat_request.messages)
        
        # Extract the latest user query
        if not conversation_history or conversation_history[-1].role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        latest_query = conversation_history[-1].content
        # Remove the latest query from history since it will be added separately
        conversation_history = conversation_history[:-1]
        
        # Build inference config from request parameters
        inference_config = {}
        if chat_request.max_tokens:
            inference_config["maxTokens"] = chat_request.max_tokens
        if chat_request.temperature is not None:
            inference_config["temperature"] = chat_request.temperature
        if chat_request.top_p is not None:
            inference_config["topP"] = chat_request.top_p
            
        # Set defaults if not provided
        if not inference_config:
            inference_config = {
                "maxTokens": 8192,
                "temperature": 0.7,
                "topP": 0.9
            }
        
        # Use provided knowledge_base_id and user_name from request if available
        kb_input = chat_request.knowledge_base_id
        user_name = chat_request.user_name

        aws_kb_id: Optional[str] = None
        project_name: Optional[str] = None

        if kb_input:
            kb_manager = get_kb_config_manager()
            kb_cfg = kb_manager.get_knowledge_base_by_id(kb_input)
            if kb_cfg and kb_cfg.get("knowledge_base_id"):
                # Caller passed internal KB id -> map to AWS KB id
                aws_kb_id = kb_cfg["knowledge_base_id"]
                project_name = kb_cfg["id"]
                logger.debug(
                    f"Translated internal KB id '{kb_input}' -> AWS knowledge_base_id '{aws_kb_id}', project='{project_name}'"
                )
            else:
                # Caller passed AWS KB id directly
                aws_kb_id = kb_input
                project_name = kb_input

        # If knowledge base parameters are provided, use knowledge base chat
        if aws_kb_id and user_name:
            try:
                kb_manager = get_kb_config_manager()
                default_settings = kb_manager.get_default_settings()

                result = chat_via_gateway_with_kb(
                    query=latest_query,
                    knowledge_base_id=aws_kb_id,
                    gateway_url=GATEWAY_URL,
                    user_name=user_name,
                    project_name=project_name,
                    system_prompt=system_prompt or default_settings.get("system_prompt"),
                    conversation_history=conversation_history,
                    inference_config=inference_config,
                    model_id=chat_request.model,
                    num_results=default_settings.get("num_results", 5),
                    search_type=default_settings.get("search_type", "HYBRID")
                )

                from uuid import uuid4
                message_id = f"chatcmpl-{str(uuid4())[:8]}"
                return convert_chat_result_to_openai_response(result, chat_request.model, message_id)
            except Exception as e:
                logger.error(f"KB chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # If no knowledge base parameters, use regular chat
        else:
            result = chat_via_gateway(
                query=latest_query,
                gateway_url=GATEWAY_URL,
                user_name=user_name or "unknown_user",
                project_name="default_project",
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                inference_config=inference_config,
                model_id=chat_request.model
            )

            from uuid import uuid4
            message_id = f"chatcmpl-{str(uuid4())[:8]}"
            return convert_chat_result_to_openai_response(result, chat_request.model, message_id)

    except Exception as e:
        logger.error(f"Error in gateway chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gateway chat error: {str(e)}")
