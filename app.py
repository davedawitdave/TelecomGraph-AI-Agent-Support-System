"""
Streamlit Web Application for RAG-powered Telecom Support Assistant.

This module provides a web-based user interface for the telecom support assistant,
allowing users to submit queries and receive AI-generated responses based on
historical conversation data.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def initialize_pipeline():
    """
    Initialize the RAG pipeline with error handling.

    Returns:
        RAGPipeline instance if successful, None if initialization fails.
    """
    try:
        from src.pipeline import RAGPipeline
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        st.error("Please check your configuration and API keys in config/secrets.yaml")
        return None


def main():
    """
    Main application function.

    Sets up the Streamlit interface and handles user interactions.
    """
    st.set_page_config(
        page_title="Telecom Support Assistant",
        page_icon="phone",
        layout="wide"
    )

    st.title("Telecom Support Assistant")
    st.markdown("*AI-powered customer support using RAG technology*")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant uses Retrieval-Augmented Generation (RAG) to provide
        intelligent responses based on historical customer support conversations.

        **Features:**
        - Semantic search through past issues
        - Context-aware response generation
        - Neo4j graph database backend
        - OpenAI GPT integration
        """)

        st.header("Setup Status")
        try:
            import yaml
            with open("config/secrets.yaml", 'r') as f:
                secrets = yaml.safe_load(f)
            api_key = secrets.get('openai', {}).get('api_key', '')
            if api_key and api_key != "your-openai-api-key-here":
                st.success("OpenAI API configured")
            else:
                st.warning("OpenAI API key not configured")

            neo4j_pass = secrets.get('neo4j', {}).get('password', '')
            if neo4j_pass and neo4j_pass != "your-neo4j-password-here":
                st.success("Neo4j configured")
            else:
                st.warning("Neo4j not configured")
        except:
            st.error("Configuration error")

    # Initialize pipeline
    pipeline = initialize_pipeline()

    if pipeline is None:
        st.error("Cannot proceed without proper configuration. Please check the setup status in the sidebar.")
        return

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Describe Your Issue")
        user_query = st.text_area(
            "Please describe your telecom problem in detail:",
            height=120,
            placeholder="Example: My internet connection is very slow and keeps dropping..."
        )

        # Advanced options
        with st.expander("Advanced Options"):
            show_similar = st.checkbox("Show similar past issues", value=False)
            max_length = st.slider("Max response length", 100, 1000, 500)

        generate_button = st.button("Get Help", type="primary", use_container_width=True)

    with col2:
        st.subheader("Quick Examples")
        examples = [
            "My internet is very slow",
            "I can't connect to WiFi",
            "My phone bill is too high",
            "I need help with mobile data",
            "My landline phone isn't working"
        ]

        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                user_query = example
                st.rerun()

    # Process query
    if generate_button and user_query.strip():
        with st.spinner("Searching for similar issues..."):
            try:
                # Get response
                response = pipeline.generate_response(user_query)

                # Show results
                st.success("Response generated successfully!")

                st.subheader("Suggested Solution")
                st.write(response)

                # Show similar issues if requested
                if show_similar:
                    with st.expander("Similar Past Issues"):
                        try:
                            similar_issues = pipeline.search.find_similar_issues(user_query, limit=3)
                            for i, issue in enumerate(similar_issues, 1):
                                st.markdown(f"**Issue {i}:** {issue.get('client_message', 'N/A')}")
                                st.markdown(f"**Solution:** {issue.get('agent_response', 'N/A')}")
                                st.markdown("---")
                        except Exception as e:
                            st.warning(f"Could not retrieve similar issues: {str(e)}")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.info("Please check your API keys and Neo4j connection.")

    elif generate_button and not user_query.strip():
        st.warning("Please describe your issue first.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, Neo4j, and OpenAI*")


if __name__ == "__main__":
    main()
