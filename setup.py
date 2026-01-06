#!/usr/bin/env python3
"""
Setup script for RAG Telecom Support Assistant
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str):
    """
    Run a command and print status.

    Args:
        command: Shell command to execute.
        description: Description of the command for logging.

    Returns:
        True if command executed successfully, False otherwise.
    """
    print(f"[SETUP] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """
    Check if Python version is compatible with the project requirements.

    Returns:
        True if Python version is compatible (3.8+), False otherwise.
    """
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"[COMPATIBLE] Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"[INCOMPATIBLE] Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def main():
    """
    Main setup function that orchestrates the installation process.
    """
    print("Setting up RAG Telecom Support Assistant")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("[ERROR] Please run this script from the project root directory (rag-project-root/)")
        return

    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return

    # Create virtual environment option
    create_venv = input("\nCreate a virtual environment? (recommended) [y/N]: ").lower().strip()
    if create_venv == 'y':
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return
        print("\n[INFO] To activate the virtual environment, run:")
        print("   source venv/bin/activate  # On Linux/Mac")
        print("   venv\\Scripts\\activate    # On Windows")

    # Check for API keys
    secrets_file = Path("config/secrets.yaml")
    if secrets_file.exists():
        print("\n[IMPORTANT] Please configure your API keys in config/secrets.yaml")
        print("   - OpenAI API key for LLM generation")
        print("   - Neo4j credentials for graph database")
    else:
        print("\n[ERROR] config/secrets.yaml not found!")

    print("\n[NEXT STEPS]")
    print("1. Set up Neo4j database (see README.md)")
    print("2. Configure API keys in config/secrets.yaml")
    print("3. Run data ingestion: python3 -c \"from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest_data()\"")
    print("4. Start the app: streamlit run app.py")

    print("\n[COMPLETE] Setup complete! Ready to build your RAG system.")

if __name__ == "__main__":
    main()
