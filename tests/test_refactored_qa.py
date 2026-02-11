#!/usr/bin/env python3
"""
Test script for the refactored QA engine with metadata and two-stage search
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.qa_engine import ProcessedDocumentLoader, QAEngine
import config

def test_metadata_extraction():
    """Test metadata extraction from filenames"""
    print("="*60)
    print("TEST 1: Metadata Extraction from Filenames")
    print("="*60)
    
    loader = ProcessedDocumentLoader()
    
    test_cases = [
        ("Q1-2025-report_text.txt", {"quarter": "Q1", "year": "2025", "doc_type": "report"}),
        ("Q4-2024-presentation_text.txt", {"quarter": "Q4", "year": "2024", "doc_type": "presentation"}),
        ("Transcript-Q4-2024-and-CMU-2025_text.txt", {"quarter": "Q4", "year": "2024", "doc_type": "transcript"}),
    ]
    
    for filename, expected in test_cases:
        metadata = loader._extract_metadata_from_filename(filename)
        print(f"\nFilename: {filename}")
        print(f"  Quarter: {metadata['quarter']} (expected: {expected['quarter']})")
        print(f"  Year: {metadata['year']} (expected: {expected['year']})")
        print(f"  Doc Type: {metadata['doc_type']} (expected: {expected['doc_type']})")
        print(f"  Title: {metadata['title']}")
        
        assert metadata['quarter'] == expected['quarter'], f"Quarter mismatch for {filename}"
        assert metadata['year'] == expected['year'], f"Year mismatch for {filename}"
        assert metadata['doc_type'] == expected['doc_type'], f"Doc type mismatch for {filename}"
    
    print("\n✅ All metadata extraction tests passed!")

def test_document_loading():
    """Test loading processed documents"""
    print("\n" + "="*60)
    print("TEST 2: Document Loading with Metadata")
    print("="*60)
    
    loader = ProcessedDocumentLoader()
    documents = loader.extract_text_from_directory(str(config.PROCESSED_DIR))
    
    if not documents:
        print("⚠️  No documents found. Make sure extractor has processed the PDFs.")
        return False
    
    print(f"\n✓ Loaded {len(documents)} document chunks")
    
    # Show sample metadata
    if documents:
        print("\nSample document metadata:")
        sample = documents[0]
        metadata = sample['metadata']
        print(f"  Source: {metadata.get('source')}")
        print(f"  Title: {metadata.get('title')}")
        print(f"  Quarter: {metadata.get('quarter')}")
        print(f"  Year: {metadata.get('year')}")
        print(f"  Doc Type: {metadata.get('doc_type')}")
        print(f"  Chunk ID: {metadata.get('chunk_id')}")
        
        # Verify all expected fields are present
        required_fields = ['source', 'title', 'doc_type', 'quarter', 'year', 'path', 'chunk_id']
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    print("\n✅ Document loading test passed!")
    return True

def test_two_stage_search():
    """Test two-stage search with document name matching"""
    print("\n" + "="*60)
    print("TEST 3: Two-Stage Search")
    print("="*60)
    
    print("\nInitializing QA engine...")
    qa_engine = QAEngine(
        data_dir=str(config.PROCESSED_DIR),
        enable_memory=False  # Disable memory for testing
    )
    
    qa_engine.load_and_index()
    
    if not qa_engine.is_indexed:
        print("⚠️  Failed to index documents")
        return False
    
    # Test 1: Query with specific quarter and year
    print("\n--- Test Query 1: 'What happened in Q4 2025?' ---")
    results = qa_engine.search_engine.search("What happened in Q4 2025?", top_k=5)
    
    print(f"Found {len(results)} results")
    for i, res in enumerate(results[:3], 1):
        metadata = res['document']['metadata']
        print(f"\n  Result {i}:")
        print(f"    Title: {metadata.get('title')}")
        print(f"    Quarter: {metadata.get('quarter')}, Year: {metadata.get('year')}")
        print(f"    Score: {res['score']:.3f} (semantic: {res['semantic_score']:.3f}, "
              f"bm25: {res['bm25_score']:.3f}, metadata: {res['metadata_score']:.3f})")
    
    # Verify Q4 2025 documents are prioritized
    top_result = results[0]['document']['metadata']
    if top_result.get('quarter') == 'Q4' and top_result.get('year') == '2025':
        print("\n  ✓ Q4 2025 document correctly prioritized!")
    else:
        print(f"\n  ⚠️  Expected Q4 2025 document, got {top_result.get('quarter')} {top_result.get('year')}")
    
    # Test 2: Query with document type
    print("\n--- Test Query 2: 'Q1 2025 transcript' ---")
    results = qa_engine.search_engine.search("Q1 2025 transcript", top_k=5)
    
    print(f"Found {len(results)} results")
    top_result = results[0]['document']['metadata']
    print(f"\n  Top result:")
    print(f"    Title: {top_result.get('title')}")
    print(f"    Doc Type: {top_result.get('doc_type')}")
    print(f"    Metadata score: {results[0]['metadata_score']:.3f}")
    
    if top_result.get('doc_type') == 'transcript' and top_result.get('quarter') == 'Q1':
        print("\n  ✓ Transcript document correctly prioritized!")
    else:
        print(f"\n  ⚠️  Expected Q1 transcript, got {top_result.get('doc_type')}")
    
    print("\n✅ Two-stage search test completed!")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("QA ENGINE REFACTORING TESTS")
    print("="*60)
    
    try:
        # Test 1: Metadata extraction (always works)
        test_metadata_extraction()
        
        # Test 2 & 3: Require processed documents
        if test_document_loading():
            test_two_stage_search()
        else:
            print("\n⚠️  Skipping search tests - no processed documents found")
            print("   Run: python scripts/extractor.py --skip-tables")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
