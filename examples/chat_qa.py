#!/usr/bin/env python3
"""
Interactive Q&A Chat with Conversation Memory

This example demonstrates the conversation memory feature that allows
the chatbot to remember previous questions and answers.

Usage:
    python example_qa_chat.py

Commands:
    - Type your question and press Enter
    - Type 'clear' to clear conversation history
    - Type 'history' to view conversation history
    - Type 'quit' or 'exit' to quit
"""

import sys
from pathlib import Path

# Add the src directory to sys.path so we can import the package
sys.path.append(str(Path(__file__).parents[1] / "src"))

from investor_relations_scraper import QAEngine, config

def print_separator():
    print("\n" + "="*70 + "\n")

def main():
    print("="*70)
    print("INVESTOR RELATIONS Q&A CHAT (with Memory)")
    print("="*70)
    print("\nInitializing Q&A engine...")
    
    # Initialize QA engine with conversation memory enabled
    qa_engine = QAEngine(
        data_dir=str(config.PROCESSED_DIR),  # Changed from RAW_DIR
        persist_directory=str(config.VECTOR_DB_DIR),
        enable_memory=True,      # Enable conversation memory
        max_history=10           # Keep last 10 Q&A pairs
    )
    
    # Load and index documents
    qa_engine.load_and_index()
    
    print("\n✓ Ready! Ask questions about Equinor's financial reports.")
    print("\nCommands: 'clear' (clear history), 'history' (view history), 'quit' (exit)")
    print_separator()
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            elif question.lower() == 'clear':
                qa_engine.clear_conversation()
                print("✓ Conversation history cleared!\n")
                continue
            
            elif question.lower() == 'history':
                history = qa_engine.get_conversation_history()
                if not history:
                    print("No conversation history yet.\n")
                else:
                    print("\n--- Conversation History ---")
                    for msg in history:
                        role = msg['role'].capitalize()
                        content = msg['content']
                        # Truncate long messages for display
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"{role}: {content}\n")
                    print("--- End of History ---\n")
                continue
            
            # Answer the question
            print("\nAssistant: ", end="", flush=True)
            result = qa_engine.answer_question(question)
            
            print(result['answer'])
            
            # Show sources
            if result.get('sources'):
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
            
            print_separator()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
