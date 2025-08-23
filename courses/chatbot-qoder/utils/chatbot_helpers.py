"""
Chatbot Helpers Module for Chatbot-Qoder Tutorial Series

This module provides utilities specific to chatbot functionality including
response generation, context management, interactive interfaces, and safety filtering.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import random
import numpy as np
from collections import deque
import json


class ConversationContext:
    """
    Manages conversation context and history for chatbots.
    
    This class maintains conversation state, including previous turns,
    user preferences, and context windows for multi-turn conversations.
    """
    
    def __init__(self, max_turns: int = 10, max_context_length: int = 512):
        """
        Initialize conversation context.
        
        Args:
            max_turns (int): Maximum number of turns to keep in history
            max_context_length (int): Maximum context length in tokens
        """
        self.max_turns = max_turns
        self.max_context_length = max_context_length
        self.history = deque(maxlen=max_turns)
        self.user_info = {}
        self.session_id = None
    
    def add_turn(self, user_input: str, bot_response: str):
        """
        Add a conversation turn to the history.
        
        Args:
            user_input (str): User's input
            bot_response (str): Bot's response
        """
        turn = {
            "user": user_input,
            "bot": bot_response,
            "timestamp": torch.tensor(0)  # Placeholder for timestamp
        }
        self.history.append(turn)
    
    def get_context_string(self, separator: str = " [SEP] ") -> str:
        """
        Get conversation history as a formatted string.
        
        Args:
            separator (str): Separator between turns
        
        Returns:
            str: Formatted context string
        """
        context_parts = []
        for turn in self.history:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Bot: {turn['bot']}")
        
        return separator.join(context_parts)
    
    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, str]]:
        """
        Get recent conversation turns.
        
        Args:
            num_turns (int): Number of recent turns to return
        
        Returns:
            List[Dict[str, str]]: Recent conversation turns
        """
        return list(self.history)[-num_turns:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
    
    def set_user_info(self, key: str, value: Any):
        """Set user information."""
        self.user_info[key] = value
    
    def get_user_info(self, key: str, default: Any = None) -> Any:
        """Get user information."""
        return self.user_info.get(key, default)


class ResponseGenerator:
    """
    Base class for response generation in chatbots.
    
    Provides common functionality for generating responses including
    text generation, filtering, and post-processing.
    """
    
    def __init__(self, 
                 max_length: int = 128,
                 min_length: int = 5,
                 temperature: float = 0.8,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2):
        """
        Initialize response generator.
        
        Args:
            max_length (int): Maximum response length
            min_length (int): Minimum response length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            repetition_penalty (float): Repetition penalty factor
        """
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
    
    def generate_response(self, 
                         model: torch.nn.Module,
                         input_ids: torch.Tensor,
                         tokenizer,
                         device: torch.device) -> str:
        """
        Generate response using the model.
        
        Args:
            model (torch.nn.Module): Language model
            input_ids (torch.Tensor): Input token IDs
            tokenizer: Tokenizer for encoding/decoding
            device (torch.device): Device to run on
        
        Returns:
            str: Generated response
        """
        model.eval()
        
        with torch.no_grad():
            generated_ids = self._generate_tokens(
                model, input_ids, device
            )
            
            # Decode generated tokens
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Post-process response
            response = self._post_process_response(response)
            
        return response
    
    def _generate_tokens(self, 
                        model: torch.nn.Module,
                        input_ids: torch.Tensor,
                        device: torch.device) -> List[int]:
        """Generate tokens using sampling strategies."""
        generated = input_ids.clone()
        
        for _ in range(self.max_length):
            # Get model predictions
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]  # Last token logits
            
            # Apply temperature
            next_token_logits = next_token_logits / self.temperature
            
            # Apply repetition penalty
            for token_id in set(generated[0].tolist()):
                next_token_logits[0, token_id] /= self.repetition_penalty
            
            # Apply top-k filtering
            if self.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, self.top_k)
                next_token_logits.fill_(-float('inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token (simplified)
            if next_token.item() == 2:  # Assuming 2 is EOS token
                break
        
        return generated[0, input_ids.size(1):].tolist()  # Return only new tokens
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response."""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response


class RuleBasedChatbot:
    """
    Simple rule-based chatbot for educational purposes.
    
    Demonstrates pattern matching, intent recognition, and template-based
    response generation commonly used in rule-based systems.
    """
    
    def __init__(self):
        """Initialize rule-based chatbot."""
        self.patterns = self._load_patterns()
        self.responses = self._load_responses()
        self.context = ConversationContext()
    
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load intent patterns."""
        return {
            "greeting": [
                r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
                r"\bhowdy\b"
            ],
            "goodbye": [
                r"\b(bye|goodbye|see you|farewell|talk to you later)\b",
                r"\bgoodnight\b"
            ],
            "how_are_you": [
                r"\bhow are you\b",
                r"\bhow're you doing\b",
                r"\bhow have you been\b"
            ],
            "what_is_name": [
                r"\bwhat('s| is) your name\b",
                r"\bwho are you\b"
            ],
            "help": [
                r"\bhelp\b",
                r"\bwhat can you do\b",
                r"\bassist\b"
            ],
            "weather": [
                r"\bweather\b",
                r"\btemperature\b",
                r"\braining\b",
                r"\bsunny\b"
            ]
        }
    
    def _load_responses(self) -> Dict[str, List[str]]:
        """Load response templates."""
        return {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! Nice to see you!",
                "Good day! How may I assist you?"
            ],
            "goodbye": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Farewell! It was nice chatting with you!",
                "Bye! Come back soon!"
            ],
            "how_are_you": [
                "I'm doing well, thank you for asking! How are you?",
                "I'm great! How about you?",
                "I'm fine, thanks! What about yourself?",
                "Doing wonderful! How are you today?"
            ],
            "what_is_name": [
                "I'm a helpful chatbot created for educational purposes!",
                "You can call me ChatBot! I'm here to help you.",
                "I'm your friendly AI assistant!",
                "I'm a chatbot designed to assist and chat with you!"
            ],
            "help": [
                "I can chat with you, answer questions, and provide assistance. What would you like to know?",
                "I'm here to help! You can ask me questions or just have a conversation.",
                "I can assist with various topics. What do you need help with?",
                "Feel free to ask me anything! I'll do my best to help."
            ],
            "weather": [
                "I don't have access to real-time weather data, but you can check a weather app or website!",
                "Sorry, I can't provide current weather information. Try checking your local weather service!",
                "I wish I could tell you about the weather, but I don't have that capability yet!"
            ],
            "default": [
                "That's interesting! Can you tell me more?",
                "I see. What else would you like to talk about?",
                "Thanks for sharing! What's on your mind?",
                "Hmm, that's something to think about. Anything else?",
                "I understand. Is there anything specific you'd like to discuss?"
            ]
        }
    
    def get_intent(self, user_input: str) -> str:
        """
        Classify user intent based on patterns.
        
        Args:
            user_input (str): User's input text
        
        Returns:
            str: Predicted intent
        """
        user_input_lower = user_input.lower()
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent
        
        return "default"
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate response for user input.
        
        Args:
            user_input (str): User's input
        
        Returns:
            str: Bot's response
        """
        intent = self.get_intent(user_input)
        responses = self.responses.get(intent, self.responses["default"])
        response = random.choice(responses)
        
        # Add to conversation context
        self.context.add_turn(user_input, response)
        
        return response


class SafetyFilter:
    """
    Safety filter for chatbot responses.
    
    Implements basic content filtering to detect and handle
    potentially harmful or inappropriate content.
    """
    
    def __init__(self):
        """Initialize safety filter."""
        self.toxic_patterns = self._load_toxic_patterns()
        self.safe_responses = self._load_safe_responses()
    
    def _load_toxic_patterns(self) -> List[str]:
        """Load patterns for detecting toxic content."""
        return [
            r"\b(hate|stupid|idiot|dumb)\b",
            r"\b(kill|die|death)\b",
            r"\b(violence|hurt|harm)\b"
            # Note: This is a simplified example
        ]
    
    def _load_safe_responses(self) -> List[str]:
        """Load safe fallback responses."""
        return [
            "I'd prefer to keep our conversation positive and helpful.",
            "Let's talk about something more constructive.",
            "I'm here to assist you in a positive way.",
            "How about we discuss something else?"
        ]
    
    def is_safe(self, text: str) -> bool:
        """
        Check if text is safe.
        
        Args:
            text (str): Text to check
        
        Returns:
            bool: True if text is safe
        """
        text_lower = text.lower()
        
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def filter_response(self, response: str) -> str:
        """
        Filter response and return safe alternative if needed.
        
        Args:
            response (str): Response to filter
        
        Returns:
            str: Safe response
        """
        if self.is_safe(response):
            return response
        else:
            return random.choice(self.safe_responses)


def interactive_chat(chatbot, max_turns: int = 50):
    """
    Interactive chat interface for testing chatbots.
    
    Args:
        chatbot: Chatbot instance with generate_response method
        max_turns (int): Maximum number of conversation turns
    
    Example:
        >>> chatbot = RuleBasedChatbot()
        >>> interactive_chat(chatbot)
    """
    print("Chatbot Interactive Interface")
    print("=" * 40)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print()
    
    safety_filter = SafetyFilter()
    turn_count = 0
    
    while turn_count < max_turns:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("Bot: Goodbye! Thanks for chatting!")
                break
            
            # Check input safety
            if not safety_filter.is_safe(user_input):
                print("Bot: " + safety_filter.filter_response(user_input))
                continue
            
            # Generate response
            response = chatbot.generate_response(user_input)
            
            # Filter response
            safe_response = safety_filter.filter_response(response)
            
            print(f"Bot: {safe_response}")
            
            turn_count += 1
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e}")


def load_conversation_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load conversation data from JSON file.
    
    Args:
        filepath (str): Path to conversation data file
    
    Returns:
        List[Dict[str, Any]]: List of conversation examples
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: Could not find conversation data file: {filepath}")
        return []


def save_conversation_log(conversation_history: List[Tuple[str, str]], 
                         filepath: str):
    """
    Save conversation history to file.
    
    Args:
        conversation_history (List[Tuple[str, str]]): Conversation turns
        filepath (str): Path to save conversation log
    """
    conversation_data = []
    for user_input, bot_response in conversation_history:
        conversation_data.append({
            "user": user_input,
            "bot": bot_response
        })
    
    with open(filepath, 'w') as f:
        json.dump(conversation_data, f, indent=2)


# Export commonly used functions and classes
__all__ = [
    "ConversationContext",
    "ResponseGenerator",
    "RuleBasedChatbot",
    "SafetyFilter",
    "interactive_chat",
    "load_conversation_data",
    "save_conversation_log"
]