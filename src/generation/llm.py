"""
LLM Response Generator using Google Gemini with RAG capabilities
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List, Union
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    Lightweight LLM Generator using Google Gemini API for customer support responses.
    Uses direct HTTP requests to the Gemini API.
    """
    
    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any]):
        """
        Initialize the LLM Generator with Gemini configuration.

        Args:
            config: Configuration dictionary containing model settings
            secrets: Secrets dictionary containing API key
            
        Raises:
            ValueError: If API key is missing
        """
        self.config = config
        self._model_name = config.get('models', {}).get('llm_model', 'gemini-1.5-flash')
        self._api_key = secrets.get('gemini', {}).get('api_key')
        
        if not self._api_key:
            raise ValueError("Gemini API key not found in secrets.yaml")
            
        self._api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}"
        self._cache: Dict[str, str] = {}
        
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a given text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _call_gemini_api(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """
        Make a direct HTTP request to the Gemini API.
        
        Args:
            prompt: The user's input prompt
            system_instruction: System message to guide the model's behavior
            
        Returns:
            Dictionary containing the response text and any sources
        """
        cache_key = self._get_cache_key(f"{prompt}:{system_instruction}")
        if cache_key in self._cache:
            return {"text": self._cache[cache_key], "sources": []}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048
            }
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(3):
            try:
                response = requests.post(
                    self._api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract response text
                cand = data.get("candidates", [{}])[0]
                text = cand.get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Cache the successful response
                self._cache[cache_key] = text
                
                return {"text": text, "sources": []}
                
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"API attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        error_msg = "Failed to get response from Gemini API after multiple attempts"
        logger.error(error_msg)
        return {"text": self._fallback_response(""), "sources": []}
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a response to a user query, optionally using RAG context.
        
        Args:
            query: The user's query
            context: Optional RAG context to augment the response
            
        Returns:
            Generated response as a string
        """
        try:
            # Create appropriate prompt based on whether we have context
            if context:
                system_instruction = """You are a helpful telecom support assistant. 
                Use the provided context to answer the user's question. 
                If the context doesn't contain the answer, say you don't know."""
                prompt = f"""Context:
{context}

Question: {query}

Answer:"""
            else:
                system_instruction = """You are a helpful telecom support assistant. 
                Provide clear and concise responses to customer queries."""
                prompt = query
            
            # Generate response
            response = self._call_gemini_api(prompt, system_instruction)
            return response["text"]
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._fallback_response(query, e)
    
    def _fallback_response(self, query: str, error: Exception = None) -> str:
        """Generate a fallback response when the main generation fails."""
        error_info = f" (Error: {str(error)})" if error else ""
        logger.error(f"Using fallback response for query: {query}{error_info}")
        
        fallbacks = [
            "I'm having trouble processing your request at the moment. Please try again in a few minutes.",
            "I apologize, but I'm experiencing some technical difficulties. Our team has been notified.",
            "I'm unable to generate a response right now. Please try again later or contact support.",
            "I'm sorry, but I'm having trouble understanding your request. Could you please rephrase it?"
        ]
        
        # Simple hash of the query to get a consistent fallback for the same query
        hash_val = hash(query) % len(fallbacks)
        return fallbacks[hash_val]
    
    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        logger.info("Response cache cleared")
