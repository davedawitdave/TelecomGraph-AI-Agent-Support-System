"""
LLM Response Generator
Generates customer support responses using Google Gemini based on retrieved similar issues
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os

class LLMGenerator:
    """
    LLM Generator using Google Gemini for customer support responses.

    This class handles the generation of contextual responses using Google's
    Gemini model, incorporating similar past issues for RAG (Retrieval-Augmented Generation).
    """

    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any]):
        """
        Initialize the LLM Generator with Gemini configuration.

        Args:
            config: Configuration dictionary containing model settings.
            secrets: Secrets dictionary containing API keys.

        Raises:
            ValueError: If Gemini API key is not found in secrets.
        """
        self.config = config
        self.model_name = config.get('models', {}).get('llm_model', 'gemini-1.5-flash')

        # Configure Gemini API
        api_key = secrets.get('gemini', {}).get('api_key')
        if not api_key:
            raise ValueError("Gemini API key not found in secrets.yaml")

        genai.configure(api_key=api_key)

        # Initialize the model with proper configuration
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception:
            # Fallback to a known working model
            print(f"Model {self.model_name} not available, trying fallback model...")
            self.model_name = "models/text-bison-001"
            self.model = genai.GenerativeModel(self.model_name)

        # Set generation configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=config.get('generation', {}).get('temperature', 0.7),
            max_output_tokens=config.get('generation', {}).get('max_tokens', 500),
        )

    def generate_response(self, query: str, similar_issues: List[Dict[str, Any]]) -> str:
        """
        Generate a response using RAG approach with Gemini.

        Args:
            query: Customer's query string.
            similar_issues: List of similar past issues and responses.

        Returns:
            Generated response string.
        """
        try:
            # Prepare context from similar issues
            context = self._prepare_context(similar_issues)

            # Create prompt
            prompt = self._create_prompt(query, context)

            # Generate response
            response = self._call_llm(prompt)

            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response(query)

    def _prepare_context(self, similar_issues: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from similar issues.

        Args:
            similar_issues: List of similar past issues.

        Returns:
            Formatted context string.
        """
        if not similar_issues:
            return "No similar issues found in the knowledge base."

        context_parts = []
        for i, issue in enumerate(similar_issues[:3]):  # Use top 3 similar issues
            context_parts.append(f"""
Similar Issue {i+1}:
Customer: {issue.get('client_message', 'N/A')}
Agent Response: {issue.get('agent_response', 'N/A')}
""")

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create the LLM prompt with context and instructions.

        Args:
            query: Customer's query.
            context: Context from similar issues.

        Returns:
            Formatted prompt string.
        """
        prompt = f"""
You are a helpful telecom customer support agent. Use the following context from similar customer issues to provide an accurate and helpful response.

CONTEXT FROM SIMILAR ISSUES:
{context}

CUSTOMER QUERY:
{query}

INSTRUCTIONS:
1. Provide a clear, helpful, and professional response
2. Base your answer on the context when relevant
3. If the context doesn't directly apply, provide general telecom support guidance
4. Keep the response concise but comprehensive
5. End with an offer to provide more assistance if needed

RESPONSE:
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the Gemini API to generate response.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated response text.

        Raises:
            Exception: If API call fails.
        """
        try:
            # Create chat session with system instruction
            chat = self.model.start_chat(history=[])

            # Add system instruction
            system_prompt = "You are a helpful telecom customer support agent."
            chat.send_message(system_prompt)

            # Send the actual prompt
            response = chat.send_message(
                prompt,
                generation_config=self.generation_config
            )

            return response.text.strip()

        except Exception as e:
            print(f"Gemini API error: {e}")
            raise

    def _fallback_response(self, query: str) -> str:
        """
        Provide a basic fallback response when API is unavailable.

        Args:
            query: Customer's original query.

        Returns:
            Fallback response string.
        """
        return f"""I understand you're experiencing an issue with: "{query}"

I'm currently unable to access our knowledge base, but here are some general troubleshooting steps you can try:

1. Restart your device
2. Check your network connection
3. Contact our support team at 1-800-HELP-NOW

Please provide more details about your issue, and I'll do my best to help you resolve it."""
