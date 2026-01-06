"""
Telecom Support Assistant with RAG and Gemini

A modern web interface for telecom support powered by RAG and Google's Gemini.
"""

import os
import sys
import time
import yaml
import streamlit as st
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after adding to path
from src import RAGPipeline

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: 500;
    }
    .stTextArea textarea {
        min-height: 120px;
    }
    .response-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4e79a7;
    }
    .similar-issue {
        background-color: #f1f3f5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_pipeline():
    """Initialize the RAG pipeline with error handling and caching."""
    try:
        # Clear any existing pipeline in session state
        if 'pipeline' in st.session_state:
            del st.session_state.pipeline
            
        # Create new pipeline
        pipeline = RAGPipeline()
        pipeline.initialize_if_needed()
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        st.error("Please check your configuration and API keys in config/secrets.yaml")
        st.exception(e)  # Show full traceback for debugging
        return None

def display_status():
    """Display system status in the sidebar."""
    # Sidebar header
    st.sidebar.header("System Status")
        
    # Check API keys and services
    try:
        with open("config/secrets.yaml", 'r') as f:
            secrets = yaml.safe_load(f)
                
        # Gemini status
        gemini_key = secrets.get('gemini', {}).get('api_key', '')
        st.sidebar.markdown("### Gemini API")
        if gemini_key and gemini_key != "your-gemini-api-key-here":
            st.sidebar.markdown("Status: Configured")
        else:
            st.sidebar.markdown("Status: Not configured")
                
        # Neo4j status
        neo4j_config = secrets.get('neo4j', {})
        st.sidebar.markdown("### Neo4j Database")
        if all(neo4j_config.get(k) for k in ['uri', 'username', 'password']):
            st.sidebar.markdown("Status: Connected")
        else:
            st.sidebar.markdown("Status: Not configured")
                
        # Vector store status
        st.sidebar.markdown("### Vector Store")
        if 'search' in st.session_state and hasattr(st.session_state.search, 'is_ready'):
            st.sidebar.markdown("Status: Ready")
        else:
            st.sidebar.markdown("Status: Initializing...")
                
    except Exception as e:
        st.sidebar.error(f"Configuration error: {str(e)}")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    if st.sidebar.button("Clear Cache", key="clear_cache_btn"):
        if 'pipeline' in st.session_state and hasattr(st.session_state.pipeline, 'generator') and hasattr(st.session_state.pipeline.generator, 'clear_cache'):
            st.session_state.pipeline.generator.clear_cache()
            st.sidebar.markdown("Cache cleared!")
            st.rerun()

def display_example_queries():
    """Display example queries in the sidebar."""
    st.sidebar.markdown("### Example Queries")
    examples = [
        "My internet is very slow",
        "I can't connect to WiFi",
        "My phone bill is too high",
        "I need help with mobile data",
        "My landline phone isn't working"
    ]
    
    for idx, example in enumerate(examples):
        if st.sidebar.button(example, key=f"example_btn_{idx}"):
            st.session_state.user_query = example
            st.rerun()

def main():
    """Main application function."""
    # Page config
    st.set_page_config(
        page_title="Telecom Support Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = initialize_pipeline()
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'response' not in st.session_state:
        st.session_state.response = None
    
    # Display sidebar
    with st.sidebar:
        st.title("Telecom Support")
        st.markdown("---")
        display_status()
        display_example_queries()
        
        # Add some space at the bottom
        st.markdown("\n\n---")
        st.markdown("*Powered by RAG and Gemini*")
    
    # Main header
    st.title("Telecom Support Assistant")
    st.markdown("Get instant help with your telecom issues using AI-powered assistance.")
    st.markdown("---")
    # Query input
    user_query = st.text_area(
        "Describe your issue:",
        height=120,
        value=st.session_state.user_query,
        placeholder="Example: My internet connection is very slow and keeps dropping..."
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            show_gemini_raw = st.checkbox("Show Gemini raw response", value=False)
        with col2:
            max_tokens = st.slider("Max response length", 100, 1000, 500)
            temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
    
    # Submit button
    if st.button("Get Help", type="primary", key="get_help_btn"):
        if not user_query.strip():
            st.warning("Please describe your issue first.")
        else:
            st.session_state.user_query = user_query
            with st.spinner("Analyzing your issue..."):
                try:
                    # Update generation config
                    st.session_state.pipeline.generator.config['generation']['max_tokens'] = max_tokens
                    st.session_state.pipeline.generator.config['generation']['temperature'] = temperature
                    
                    # Generate response
                    start_time = time.time()
                    st.session_state.response = st.session_state.pipeline.generate_response(user_query)
                    st.session_state.response_time = time.time() - start_time
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error("Please check your API keys and service connections.")
    
    # Display results
    if st.session_state.response:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Response", "Raw Gemini", "Vector Search"])
        
        with tab1:
            st.markdown("### Suggested Solution")
            st.markdown(st.session_state.response.get('rag_response', 'No response generated'))
        
        with tab2:
            st.markdown("### Raw Gemini Response")
            st.code(st.session_state.response.get('gemini_response', 'No raw response available'), language='text')
        
        with tab3:
            st.markdown("### Similar Past Issues")
            if 'vector_search_results' in st.session_state.response and st.session_state.response['vector_search_results']:
                for i, result in enumerate(st.session_state.response['vector_search_results'][:3]):
                    with st.expander(f"Similar Issue {i+1} (Relevance: {result.get('score', 0):.2f})"):
                        st.markdown(f"**Customer Issue:**\n{result.get('client_message', 'N/A')}")
                        st.markdown(f"\n**Agent Response:**\n{result.get('agent_response', 'N/A')}")
                        
                        # Show additional metadata if available
                        if 'category' in result or 'resolution_time' in result or 'satisfaction' in result:
                            st.markdown("\n**Additional Info:**")
                            cols = st.columns(3)
                            if 'category' in result:
                                cols[0].markdown(f"**Category:**\n{result['category']}")
                            if 'resolution_time' in result:
                                cols[1].markdown(f"**Resolution Time:**\n{result['resolution_time']}")
                            if 'satisfaction' in result:
                                cols[2].markdown(f"**Satisfaction:**\n{result['satisfaction']}")
            else:
                st.markdown("No similar issues found.")
        
        # Show performance metrics
        with st.expander("Performance Metrics"):
            if hasattr(st.session_state, 'response_time'):
                st.markdown("#### Processing Time")
                
                # Create columns for metrics
                if 'vector_search_results' in st.session_state.response:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Time", f"{st.session_state.response_time:.2f}s")
                    col2.metric("Similar Issues", len(st.session_state.response['vector_search_results']))
                    col3.metric("Response Length", f"{len(st.session_state.response.get('rag_response', ''))} chars")
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("Total Time", f"{st.session_state.response_time:.2f}s")
                    col2.metric("Response Length", f"{len(st.session_state.response.get('rag_response', ''))} chars")
    
    # Add some space at the bottom
    st.markdown("\n\n---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>Telecom Support Assistant v1.0 | Built with RAG and Gemini</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
