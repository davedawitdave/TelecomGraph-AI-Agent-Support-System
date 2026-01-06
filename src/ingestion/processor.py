"""
Data Processor for Telecom Conversations
Extracts (Client_Msg, Agent_Response) pairs from the dataset
"""

from typing import List, Dict, Any, Tuple
import re

class DataProcessor:
    def __init__(self):
        self.conversation_pairs = []

    def process_conversations(self, dataset) -> List[Dict[str, Any]]:
        """
        Process the dataset and extract conversation pairs.

        Args:
            dataset: Dataset object (can be a split or full dataset)

        Returns:
            List of conversation pairs with client messages and agent responses
        """
        processed_data = []

        # Handle both dataset splits and direct datasets
        if hasattr(dataset, 'column_names'):
            # Direct dataset (arrow dataset)
            data_source = dataset
        elif 'train' in dataset:
            # Dataset dict with splits
            data_source = dataset['train']
        else:
            # Assume it's iterable
            data_source = dataset

        # Group messages by conversation_id
        conversations = {}
        for item in data_source:
            conv_id = item['conversation_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(item)

        # Extract pairs from each conversation
        for conv_id, messages in conversations.items():
            pairs = self.extract_conversation_pairs_from_messages(conv_id, messages)
            processed_data.extend(pairs)

        print(f"Extracted {len(processed_data)} conversation pairs from {len(conversations)} conversations")
        return processed_data

    def extract_conversation_pairs_from_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract client-agent message pairs from a list of messages in a conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries with speaker, text, etc.

        Returns:
            List of conversation pairs
        """
        pairs = []

        # Sort messages by timestamp if available
        if messages and 'date_time' in messages[0]:
            messages.sort(key=lambda x: x.get('date_time', ''))

        # Extract client-agent pairs
        i = 0
        while i < len(messages) - 1:
            current_msg = messages[i]
            next_msg = messages[i + 1]

            # Check if we have client followed by agent
            if (current_msg.get('speaker') == 'client' and
                next_msg.get('speaker') == 'agent'):

                pair = {
                    'client_message': current_msg.get('text', '').strip(),
                    'agent_response': next_msg.get('text', '').strip(),
                    'conversation_id': conversation_id,
                    'metadata': {
                        'client_time': current_msg.get('date_time'),
                        'agent_time': next_msg.get('date_time')
                    }
                }

                # Only add if both messages have content
                if pair['client_message'] and pair['agent_response']:
                    pairs.append(pair)

                # Skip the agent message since it's been paired
                i += 2
            else:
                # Move to next message
                i += 1

        return pairs

    def split_conversation_turns(self, conversation_text: str) -> List[Dict[str, str]]:
        """Split conversation text into individual turns"""
        # This is a placeholder - actual implementation depends on dataset format
        # You might need to parse different formats like:
        # "Client: Hello\nAgent: Hi there\nClient: I have an issue..."

        turns = []
        lines = conversation_text.split('\n')

        for line in lines:
            if ': ' in line:
                speaker, message = line.split(': ', 1)
                turns.append({
                    'speaker': speaker.lower().strip(),
                    'message': message.strip()
                })

        return turns

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Add more cleaning logic as needed
        return text
