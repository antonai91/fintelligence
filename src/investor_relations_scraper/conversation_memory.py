"""
Conversation Memory

Manages conversation history for the chat interface with
optional persistence to disk.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional


class ConversationMemory:
    """Manages conversation history for the chat interface"""
    
    def __init__(self, max_messages: int = 10, persist_path: Optional[str] = None):
        """
        Initialize conversation memory
        
        Args:
            max_messages: Maximum number of message pairs to keep in memory
            persist_path: Optional path to save/load conversation history
        """
        self.max_messages = max_messages
        self.messages = []  # List of {"role": "user"/"assistant", "content": "..."}
        self.persist_path = Path(persist_path) if persist_path else None
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only the last max_messages pairs (user + assistant = 2 messages per pair)
        max_total = self.max_messages * 2
        if len(self.messages) > max_total:
            self.messages = self.messages[-max_total:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.messages.copy()
    
    def get_formatted_history(self) -> str:
        """Get conversation history as a formatted string"""
        if not self.messages:
            return ""
        
        formatted = "Previous conversation:\n"
        for msg in self.messages:
            role = msg["role"].capitalize()
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted
    
    def clear(self):
        """Clear all conversation history"""
        self.messages = []
    
    def save(self):
        """Save conversation history to disk"""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.messages, f)
            print(f"Saved conversation history to {self.persist_path}")
    
    def load(self) -> bool:
        """Load conversation history from disk"""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, 'rb') as f:
                    self.messages = pickle.load(f)
                print(f"Loaded conversation history with {len(self.messages)} messages")
                return True
            except Exception as e:
                print(f"Error loading conversation history: {e}")
        return False
