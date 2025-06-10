import logging
import requests
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def get_retrieve_config(
    query: str,
    knowledge_base_id: str,
    gateway_url: str,
    user_name: str = "unknown_user",
    project_name: str = "unknown_project",
    num_results: int = 5,
    search_type: str = "HYBRID"
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Retrieve information from a knowledge base.
    
    Args:
        query: The search query
        knowledge_base_id: ID of the knowledge base to search
        gateway_url: API Gateway URL
        user_name: User identifier
        project_name: Project identifier
        num_results: Number of results to retrieve
        search_type: Type of search (HYBRID, VECTOR, KEYWORD)
    
    Returns:
        Tuple of (source_name, text_chunks, metadata)
    """
    try:
        # Validate required parameters
        if not query:
            logger.error("Query parameter is empty or None")
            return "unknown", [], {"error": "Empty query"}
        
        if not knowledge_base_id:
            logger.error("Knowledge base ID is empty or None")
            return "unknown", [], {"error": "Empty knowledge_base_id"}
        
        logger.info(f"Retrieving from KB '{knowledge_base_id}' with query: '{query[:50]}...'")
        
        # Prepare the retrieval payload - using correct field names from documentation
        payload = {
            "action": "retrieve",
            "payload": {
                "query": query,
                "knowledgeBaseId": knowledge_base_id,
                "retrievalConfig": {
                    "vectorSearchConfiguration": {
                        "overrideSearchType": search_type,
                        "numberOfResults": num_results
                    }
                },
                "user": user_name,
                "project": project_name
            }
        }
        
        # Make the API call
        import os
        api_key = os.getenv('BEDROCK_API_KEY')
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        
        response = requests.post(gateway_url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Retrieval failed with status {response.status_code}: {response.text}")
            logger.error(f"Failed payload was: {payload}")
            return "unknown", [], {}
        
        data = response.json()
        
        # Extract the retrieved chunks with their sources
        retrieved_results = data.get("retrievalResults", [])
        text_chunks = []
        sources = []
        
        for result in retrieved_results:
            # The text is directly in the result, not nested under 'content'
            text = result.get("text", "")
            source = result.get("source", "")
            if text:
                text_chunks.append(text)
                sources.append(source)
        
        source_name = "knowledge_base"
        metadata = {
            "num_results": len(text_chunks),
            "search_type": search_type,
            "sources": sources
        }
        
        logger.info(f"Retrieved {len(text_chunks)} chunks from {source_name}")
        return source_name, text_chunks, metadata
        
    except Exception as e:
        logger.error(f"Error in get_retrieve_config: {str(e)}")
        return "unknown", [], {"error": str(e)} 