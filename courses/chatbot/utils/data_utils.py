"""
Data Utilities for PyTorch Chatbot Tutorial

This module provides educational data handling utilities for the chatbot tutorial series.
All functions include comprehensive documentation and explanations to help learners
understand data preprocessing concepts in NLP and machine learning.

Educational Focus:
    - Clear explanations of data preprocessing steps
    - Detailed comments on NLP-specific operations
    - Examples of common data handling patterns
    - Debugging tips and common pitfalls
"""

import json
import os
import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """
    Educational data model for individual messages within conversations.
    
    This class demonstrates how to structure conversational data for ML training.
    Each message contains the essential information needed for chatbot training.
    
    Educational Notes:
        - Speaker identification helps distinguish between different participants
        - Text preprocessing is crucial for consistent model input
        - Timestamps can be useful for conversation flow analysis
    """
    speaker: str
    text: str
    timestamp: Optional[datetime] = None
    
    def preprocess(self) -> str:
        """
        Preprocess message text for training.
        
        Educational Purpose:
            Demonstrates common text preprocessing steps used in NLP:
            - Lowercasing for consistency
            - Removing extra whitespace
            - Basic punctuation handling
        
        Returns:
            str: Preprocessed text ready for tokenization
            
        Learning Notes:
            - Preprocessing choices affect model performance
            - Different tasks may require different preprocessing strategies
            - Balance between cleaning and preserving meaning is important
        """
        # Convert to lowercase for consistency
        text = self.text.lower()
        
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic punctuation normalization (educational example)
        text = re.sub(r'[^\w\s\.\?\!,]', '', text)
        
        return text


@dataclass
class Conversation:
    """
    Educational data model for conversation representation.
    
    This class shows how to structure multi-turn conversations for chatbot training.
    It demonstrates the relationship between messages and how to extract training pairs.
    
    Educational Notes:
        - Conversations are sequences of messages between participants
        - Context helps maintain conversation coherence
        - Metadata can store additional information for analysis
    """
    id: str
    messages: List[Message]
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_training_pairs(self) -> List[Tuple[str, str]]:
        """
        Convert conversation to input-output pairs for training.
        
        Educational Purpose:
            Demonstrates how to extract training data from conversations:
            - Each message becomes a response to the previous message
            - This creates input-output pairs for supervised learning
        
        Returns:
            List[Tuple[str, str]]: List of (input, output) pairs
            
        Learning Notes:
            - Training pairs are the foundation of supervised chatbot training
            - Quality of pairs directly affects model performance
            - Consider conversation context and flow when creating pairs
        """
        pairs = []
        
        # Extract consecutive message pairs for training
        for i in range(len(self.messages) - 1):
            input_msg = self.messages[i].preprocess()
            output_msg = self.messages[i + 1].preprocess()
            
            # Only create pairs where speakers are different (actual conversation)
            if self.messages[i].speaker != self.messages[i + 1].speaker:
                pairs.append((input_msg, output_msg))
        
        return pairs


class ChatbotDataset(Dataset):
    """
    Educational PyTorch Dataset for chatbot training.
    
    This class demonstrates how to create custom PyTorch datasets for NLP tasks.
    It handles conversation data and prepares it for model training with detailed
    explanations of each step.
    
    Educational Purpose:
        - Shows PyTorch Dataset patterns and best practices
        - Demonstrates tokenization integration with datasets
        - Explains data loading concepts for sequence-to-sequence tasks
    """
    
    def __init__(self, conversations: List[Conversation], tokenizer, max_length: int = 512):
        """
        Initialize the chatbot dataset.
        
        Args:
            conversations: List of Conversation objects
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length for padding/truncation
            
        Educational Notes:
            - Dataset initialization prepares data for efficient loading
            - Tokenizer integration happens at dataset level for consistency
            - Max length prevents memory issues with variable-length sequences
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract all training pairs from conversations
        self.training_pairs = []
        for conv in conversations:
            pairs = conv.to_training_pairs()
            self.training_pairs.extend(pairs)
        
        print(f"Educational Dataset Info:")
        print(f"- Total conversations: {len(conversations)}")
        print(f"- Total training pairs: {len(self.training_pairs)}")
        print(f"- Max sequence length: {max_length}")
    
    def __len__(self) -> int:
        """
        Return the number of training pairs in the dataset.
        
        Educational Notes:
            - PyTorch requires __len__ method for datasets
            - This determines how many batches will be created during training
            - Length affects training time and epoch definition
        """
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example by index.
        
        Args:
            idx: Index of the training pair to retrieve
            
        Returns:
            Dict containing tokenized input and target sequences
            
        Educational Purpose:
            - Demonstrates PyTorch's data loading mechanism
            - Shows how to tokenize and prepare text for model input
            - Explains padding and attention mask concepts
        """
        input_text, target_text = self.training_pairs[idx]
        
        # Tokenize input and target sequences
        # Educational Note: Tokenization converts text to numerical representations
        input_encoding = self.tokenizer.encode(input_text, max_length=self.max_length)
        target_encoding = self.tokenizer.encode(target_text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(input_encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(input_encoding['attention_mask'], dtype=torch.long),
            'target_ids': torch.tensor(target_encoding['input_ids'], dtype=torch.long),
            'target_attention_mask': torch.tensor(target_encoding['attention_mask'], dtype=torch.long)
        }


def load_conversations_from_json(file_path: str) -> List[Conversation]:
    """
    Load conversations from a JSON file with educational error handling.
    
    Args:
        file_path: Path to the JSON file containing conversation data
        
    Returns:
        List[Conversation]: Loaded conversations
        
    Educational Purpose:
        - Demonstrates file I/O for ML data loading
        - Shows proper error handling for data loading operations
        - Explains JSON data structure for conversations
        
    Expected JSON Format:
        [
            {
                "id": "conv_1",
                "messages": [
                    {"speaker": "user", "text": "Hello"},
                    {"speaker": "bot", "text": "Hi there!"}
                ],
                "context": "greeting",
                "metadata": {}
            }
        ]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        for conv_data in data:
            # Create Message objects from JSON data
            messages = []
            for msg_data in conv_data['messages']:
                message = Message(
                    speaker=msg_data['speaker'],
                    text=msg_data['text'],
                    timestamp=msg_data.get('timestamp')
                )
                messages.append(message)
            
            # Create Conversation object
            conversation = Conversation(
                id=conv_data['id'],
                messages=messages,
                context=conv_data.get('context'),
                metadata=conv_data.get('metadata', {})
            )
            conversations.append(conversation)
        
        print(f"Educational Data Loading Success:")
        print(f"- Loaded {len(conversations)} conversations from {file_path}")
        
        return conversations
        
    except FileNotFoundError as e:
        print(f"Educational Error: File not found - {e}")
        print("Learning Note: This error occurs when the specified file doesn't exist")
        print("Solution: Check file path and ensure data file exists")
        raise
        
    except json.JSONDecodeError as e:
        print(f"Educational Error: Invalid JSON format - {e}")
        print("Learning Note: JSON parsing errors indicate malformed data structure")
        print("Debugging Tip: Validate JSON format using online validators")
        raise
        
    except KeyError as e:
        print(f"Educational Error: Missing required field - {e}")
        print("Learning Note: Data structure doesn't match expected format")
        print("Solution: Ensure JSON contains required fields (id, messages)")
        raise


def create_simple_qa_dataset(output_path: str) -> None:
    """
    Create a simple Q&A dataset for educational purposes.
    
    Args:
        output_path: Path where the JSON dataset will be saved
        
    Educational Purpose:
        - Demonstrates how to create training data for chatbots
        - Shows the structure needed for conversation datasets
        - Provides a starting point for experimentation
        
    Learning Notes:
        - Simple datasets help understand model behavior
        - Quality over quantity in educational contexts
        - Diverse examples improve model generalization
    """
    simple_conversations = [
        {
            "id": "conv_001",
            "messages": [
                {"speaker": "user", "text": "Hello"},
                {"speaker": "bot", "text": "Hi there! How can I help you today?"}
            ],
            "context": "greeting",
            "metadata": {"difficulty": "easy", "category": "greeting"}
        },
        {
            "id": "conv_002", 
            "messages": [
                {"speaker": "user", "text": "What is machine learning?"},
                {"speaker": "bot", "text": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."}
            ],
            "context": "educational",
            "metadata": {"difficulty": "medium", "category": "education"}
        },
        {
            "id": "conv_003",
            "messages": [
                {"speaker": "user", "text": "How are you?"},
                {"speaker": "bot", "text": "I'm doing well, thank you for asking! How are you doing?"}
            ],
            "context": "casual",
            "metadata": {"difficulty": "easy", "category": "casual"}
        },
        {
            "id": "conv_004",
            "messages": [
                {"speaker": "user", "text": "Can you help me with Python?"},
                {"speaker": "bot", "text": "Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?"}
            ],
            "context": "programming",
            "metadata": {"difficulty": "medium", "category": "programming"}
        }
    ]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Educational Dataset Created:")
    print(f"- Simple Q&A dataset saved to: {output_path}")
    print(f"- Contains {len(simple_conversations)} conversations")
    print(f"- Ready for use in chatbot training tutorials")


def validate_dataset(conversations: List[Conversation]) -> Dict[str, Any]:
    """
    Validate dataset quality and provide educational insights.
    
    Args:
        conversations: List of conversations to validate
        
    Returns:
        Dict containing validation results and statistics
        
    Educational Purpose:
        - Demonstrates data quality assessment techniques
        - Shows important metrics for conversation datasets
        - Helps identify potential training issues early
    """
    stats = {
        'total_conversations': len(conversations),
        'total_messages': 0,
        'total_training_pairs': 0,
        'avg_messages_per_conversation': 0,
        'speakers': set(),
        'empty_messages': 0,
        'validation_passed': True,
        'issues': []
    }
    
    for conv in conversations:
        stats['total_messages'] += len(conv.messages)
        stats['total_training_pairs'] += len(conv.to_training_pairs())
        
        for msg in conv.messages:
            stats['speakers'].add(msg.speaker)
            if not msg.text.strip():
                stats['empty_messages'] += 1
                stats['issues'].append(f"Empty message in conversation {conv.id}")
    
    # Calculate averages
    if stats['total_conversations'] > 0:
        stats['avg_messages_per_conversation'] = stats['total_messages'] / stats['total_conversations']
    
    # Validation checks
    if stats['empty_messages'] > 0:
        stats['validation_passed'] = False
    
    if len(stats['speakers']) < 2:
        stats['validation_passed'] = False
        stats['issues'].append("Dataset should contain at least 2 different speakers")
    
    # Educational output
    print(f"Educational Dataset Validation:")
    print(f"- Total conversations: {stats['total_conversations']}")
    print(f"- Total messages: {stats['total_messages']}")
    print(f"- Training pairs: {stats['total_training_pairs']}")
    print(f"- Average messages per conversation: {stats['avg_messages_per_conversation']:.2f}")
    print(f"- Unique speakers: {len(stats['speakers'])}")
    print(f"- Validation passed: {stats['validation_passed']}")
    
    if stats['issues']:
        print("Issues found:")
        for issue in stats['issues']:
            print(f"  - {issue}")
    
    return stats