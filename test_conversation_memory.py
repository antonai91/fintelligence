#!/usr/bin/env python3
"""
Test script to verify conversation memory functionality
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.qa_engine import ConversationMemory

def test_conversation_memory():
    """Test the ConversationMemory class"""
    
    print("Testing ConversationMemory class...")
    print("="*50)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = Path(tmpdir) / "test_conversation.pkl"
        
        # Test 1: Basic functionality
        print("\n1. Testing basic add/get functionality...")
        memory = ConversationMemory(max_messages=3, persist_path=str(persist_path))
        
        memory.add_message("user", "What is the revenue?")
        memory.add_message("assistant", "The revenue is $100M")
        memory.add_message("user", "What about profit?")
        memory.add_message("assistant", "The profit is $20M")
        
        history = memory.get_history()
        assert len(history) == 4, f"Expected 4 messages, got {len(history)}"
        print(f"   ✓ Added 4 messages successfully")
        
        # Test 2: Max messages limit
        print("\n2. Testing max_messages limit (3 pairs = 6 messages)...")
        memory.add_message("user", "What about expenses?")
        memory.add_message("assistant", "Expenses are $80M")
        memory.add_message("user", "What about growth?")
        memory.add_message("assistant", "Growth is 10%")
        
        history = memory.get_history()
        assert len(history) == 6, f"Expected 6 messages (max), got {len(history)}"
        assert history[0]["content"] == "What about profit?", "Oldest messages should be removed"
        print(f"   ✓ Correctly limited to {len(history)} messages")
        
        # Test 3: Formatted history
        print("\n3. Testing formatted history...")
        formatted = memory.get_formatted_history()
        assert "Previous conversation:" in formatted
        assert "User:" in formatted
        assert "Assistant:" in formatted
        print("   ✓ Formatted history works correctly")
        
        # Test 4: Save and load
        print("\n4. Testing save/load functionality...")
        memory.save()
        assert persist_path.exists(), "Persist file should exist"
        
        # Create new memory instance and load
        memory2 = ConversationMemory(max_messages=3, persist_path=str(persist_path))
        loaded = memory2.load()
        assert loaded, "Load should succeed"
        assert len(memory2.get_history()) == 6, "Loaded history should match saved"
        print("   ✓ Save/load works correctly")
        
        # Test 5: Clear functionality
        print("\n5. Testing clear functionality...")
        memory2.clear()
        assert len(memory2.get_history()) == 0, "History should be empty after clear"
        print("   ✓ Clear works correctly")
    
    print("\n" + "="*50)
    print("✅ All tests passed!")

if __name__ == "__main__":
    try:
        test_conversation_memory()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
