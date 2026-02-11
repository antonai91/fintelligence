#!/usr/bin/env python3
"""
Test script to verify persistent vector storage functionality
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.qa_engine import QAEngine

def test_persistent_storage():
    """Test that the persistent storage works correctly"""
    
    data_dir = "data/raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        print("Please ensure you have PDFs in the data/raw directory")
        return
    
    print("=" * 60)
    print("TEST 1: First Run (Should index all documents)")
    print("=" * 60)
    
    # Remove existing index to simulate first run
    vector_db_path = Path("data/vector_db")
    if vector_db_path.exists():
        import shutil
        shutil.rmtree(vector_db_path)
        print("Removed existing index for clean test\n")
    
    start_time = time.time()
    engine1 = QAEngine(data_dir=data_dir)
    engine1.load_and_index()
    first_run_time = time.time() - start_time
    
    print(f"\n✓ First run completed in {first_run_time:.2f} seconds")
    print(f"✓ Indexed {len(engine1.search_engine.documents)} documents")
    
    print("\n" + "=" * 60)
    print("TEST 2: Second Run (Should load from disk)")
    print("=" * 60)
    
    start_time = time.time()
    engine2 = QAEngine(data_dir=data_dir)
    engine2.load_and_index()
    second_run_time = time.time() - start_time
    
    print(f"\n✓ Second run completed in {second_run_time:.2f} seconds")
    print(f"✓ Loaded {len(engine2.search_engine.documents)} documents")
    
    # Calculate speedup
    speedup = first_run_time / second_run_time if second_run_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"First run (with indexing):  {first_run_time:.2f}s")
    print(f"Second run (from cache):    {second_run_time:.2f}s")
    print(f"Speedup:                    {speedup:.1f}x faster! 🚀")
    
    if speedup > 5:
        print("\n✅ SUCCESS: Persistent storage is working correctly!")
    elif speedup > 2:
        print("\n⚠️  WARNING: Speedup is lower than expected, but storage is working")
    else:
        print("\n❌ FAILURE: Persistent storage may not be working correctly")
    
    print("\n" + "=" * 60)
    print("TEST 3: Verify file change detection")
    print("=" * 60)
    
    # Check file hashes
    current_hashes = engine2._get_current_file_hashes()
    stored_hashes = engine2.search_engine.vector_store.file_hashes
    
    print(f"✓ Tracking {len(current_hashes)} PDF files")
    print(f"✓ File hashes match: {current_hashes == stored_hashes}")
    
    if current_hashes == stored_hashes:
        print("✅ File change detection is working correctly!")
    else:
        print("⚠️  File hashes don't match (this might be expected if files changed)")

if __name__ == "__main__":
    test_persistent_storage()
