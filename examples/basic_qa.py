
import os
import sys
from pathlib import Path

# Add the src directory to sys.path so we can import the package
sys.path.append(str(Path(__file__).parents[1] / "src"))

from investor_relations_scraper import QAEngine, config

def main():
    # Check if OPENAI_API_KEY is set (config.py loads from .env automatically)
    try:
        api_key = config.get_openai_api_key()
        print(f"✓ API Key loaded successfully")
    except ValueError as e:
        print(f"❌ {e}")
        print("\nPlease ensure your .env file contains:")
        print("OPENAI_API_KEY=your-actual-api-key-here")
        return

    # Initialize QA Engine
    # Pointing to the processed directory where cleaned text files are stored
    # Using config for directory paths
    data_dir = str(config.PROCESSED_DIR)  # Changed from RAW_DIR
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run the extractor first to process PDFs:")
        print("  python scripts/extractor.py --skip-tables")
        return
        
    print(f"\n📚 Initializing QA Engine with data directory: {data_dir}")
    print(f"🤖 Using model: {config.MODEL_QA}")
    print(f"🔢 Using embedding model: {config.MODEL_EMBEDDING}\n")
    
    engine = QAEngine(data_dir=data_dir)
    
    # Load and Index (this happens automatically on first query, but explicit here for demo)
    engine.load_and_index()
    
    # Interactive loop
    print("\n" + "="*50)
    print("Equinor Investor Relations QA System")
    print("Ask a question about the financial reports (type 'exit' to quit)")
    print("="*50 + "\n")
    
    while True:
        question = input("\nYour Question: ")
        if question.lower() in ('exit', 'quit', 'q'):
            break
            
        if not question.strip():
            continue
            
        response = engine.answer_question(question)
        
        print("\n" + "-"*30)
        print("ANSWER:")
        print(response["answer"])
        print("-"*30)
        print(f"Sources used: {', '.join(response['sources'])}")
        
        # Optional: Print debug info about retrieval scores
        # for i, chunk in enumerate(response['retrieved_chunks']):
        #     print(f"Debug - Chunk {i+1}: Score {chunk['score']:.3f} ({chunk['document']['metadata']['source']})")

if __name__ == "__main__":
    main()
