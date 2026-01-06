"""
Data Loader Module

Handles loading and management of the telecom conversation dataset.
Supports both local dataset storage and fallback to Hugging Face download.
"""

from datasets import load_dataset, load_from_disk
from typing import Dict, Any, Optional
import os
from pathlib import Path


class DataLoader:
    """
    Data loader for telecom conversation corpus.

    This class manages the loading of the telecom conversation dataset,
    prioritizing local storage over network downloads for efficiency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader.

        Args:
            config: Configuration dictionary containing data settings.
        """
        self.config = config
        self.dataset_name = "eisenzopf/telecom-conversation-corpus"
        self.cache_dir = config.get('data', {}).get('cache_dir', 'data/01_cache')
        self.local_dataset_path = Path("data/telecom_conversation_corpus")

    def load_dataset(self) -> Dict[str, Any]:
        """
        Load the telecom conversation dataset.

        First attempts to load from local storage, then falls back to
        downloading from Hugging Face if local copy doesn't exist.

        Returns:
            Dataset dictionary containing train/validation/test splits.

        Raises:
            Exception: If dataset loading fails from both local and remote sources.
        """
        # Try loading from local storage first
        if self._is_local_dataset_available():
            print("Loading dataset from local storage...")
            try:
                dataset = load_from_disk(str(self.local_dataset_path))
                print(f"Dataset loaded from local storage. Available splits: {list(dataset.keys())}")
                for split_name, split_data in dataset.items():
                    print(f"  {split_name}: {len(split_data)} samples")
                return dataset
            except Exception as e:
                print(f"Failed to load local dataset: {e}")

        # Fallback to Hugging Face download
        print(f"Downloading dataset from Hugging Face: {self.dataset_name}")
        try:
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir
            )
            print(f"Dataset downloaded successfully. Train size: {len(dataset['train'])}")

            # Save to local storage for future use
            self._save_dataset_locally(dataset)
            return dataset

        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            raise

    def _is_local_dataset_available(self) -> bool:
        """
        Check if the dataset is available in local storage.

        Returns:
            True if local dataset exists and appears valid, False otherwise.
        """
        if not self.local_dataset_path.exists():
            return False

        # Check for essential dataset files - either dataset_dict.json (new format) or dataset_info.json (old format)
        has_root_config = (
            (self.local_dataset_path / 'dataset_dict.json').exists() or
            (self.local_dataset_path / 'dataset_info.json').exists()
        )

        if not has_root_config:
            return False

        # Check for train directory and its files
        train_dir = self.local_dataset_path / 'train'
        if not train_dir.exists():
            return False

        # Check for essential train files
        train_config_files = ['dataset_info.json', 'state.json']
        for file_path in train_config_files:
            if not (train_dir / file_path).exists():
                return False

        # Check for at least one data file
        data_files = list(train_dir.glob('data-*.arrow'))
        if not data_files:
            return False

        return True

    def _save_dataset_locally(self, dataset: Dict[str, Any]) -> None:
        """
        Save the dataset to local storage for future use.

        Args:
            dataset: Dataset dictionary to save.
        """
        try:
            print("Saving dataset to local storage...")
            self.local_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self.local_dataset_path))
            print(f"Dataset saved to {self.local_dataset_path}")
        except Exception as e:
            print(f"Warning: Failed to save dataset locally: {e}")

    def get_sample_data(self, n_samples: int = 100) -> Any:
        """
        Get a sample of the dataset for testing purposes.

        Args:
            n_samples: Number of samples to retrieve.

        Returns:
            Sample dataset split.
        """
        dataset = self.load_dataset()
        train_data = dataset['train']
        return train_data.select(range(min(n_samples, len(train_data))))
