"""
Search Engine Components

Provides persistent vector storage (FAISS) and hybrid search
combining semantic similarity with BM25 keyword matching.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from . import config


class PersistentVectorStore:
    """Stores embeddings and text chunks using FAISS with pickle persistence"""
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_directory / "faiss.index"
        self.documents_path = self.persist_directory / "documents.pkl"
        self.metadata_path = self.persist_directory / "metadata.pkl"
        
        self.documents = []
        self.index = None
        self.file_hashes = {}
        
    def load(self) -> bool:
        """Load existing index from disk"""
        try:
            if self.index_path.exists() and self.documents_path.exists():
                print("Loading existing vector index from disk...")
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'rb') as f:
                        self.file_hashes = pickle.load(f)
                
                print(f"Loaded {len(self.documents)} documents from disk")
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
        
        return False
        
    def save(self):
        """Save index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.file_hashes, f)
            
            print(f"Saved index with {len(self.documents)} documents to disk")
        
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray, file_hashes: Dict[str, str]):
        """Add documents and their embeddings to the store"""
        self.documents = documents
        self.file_hashes = file_hashes
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.save()
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS
        """
        if self.index is None or len(self.documents) == 0:
            return []
            
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "type": "semantic"
                })
            
        return results


class HybridSearchEngine:
    """Combines semantic search and BM25 keyword search"""
    
    def __init__(self, model_name: Optional[str] = None, persist_directory: Optional[str] = None):
        # Use config defaults if not specified
        model_name = model_name or config.MODEL_EMBEDDING
        persist_directory = persist_directory or str(config.VECTOR_DB_DIR)
        device = config.MODEL_DEVICE
        
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.bm25 = None
        self.bm25_path = Path(persist_directory) / "bm25.pkl"
        self.vector_store = PersistentVectorStore(persist_directory)
        self.documents = []
        
    def load_existing_index(self) -> bool:
        """Try to load existing index from disk"""
        if self.vector_store.load():
            self.documents = self.vector_store.documents
            
            # Load BM25 index if it exists
            if self.bm25_path.exists():
                try:
                    with open(self.bm25_path, 'rb') as f:
                        self.bm25 = pickle.load(f)
                    print("Loaded BM25 index from disk")
                except Exception as e:
                    print(f"Error loading BM25 index: {e}")
            
            return True
        return False
        
    def index_documents(self, documents: List[Dict[str, Any]], file_hashes: Dict[str, str]):
        """Index documents for both semantic and keyword search"""
        self.documents = documents
        
        if not documents:
            print("No documents to index.")
            return

        # 1. Prepare Semantic Search
        print("Generating embeddings (this may take a while)...")
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=config.EMBEDDING_BATCH_SIZE
        )
        self.vector_store.add_documents(documents, embeddings, file_hashes)
        
        # 2. Prepare BM25 Search
        print("Building BM25 index...")
        tokenized_corpus = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        print("Saved BM25 index to disk")
        
    def _search_document_metadata(self, query: str) -> Dict[int, float]:
        """
        Search document metadata (title, quarter, year, doc_type)
        Returns dict of document index -> relevance score
        """
        query_lower = query.lower()
        metadata_scores = {}
        
        for i, doc in enumerate(self.documents):
            score = 0.0
            metadata = doc.get("metadata", {})
            
            # Check quarter match (e.g., "Q4", "Q1")
            if metadata.get("quarter"):
                quarter = metadata["quarter"]
                if quarter.lower() in query_lower:
                    score += 0.4
            
            # Check year match (e.g., "2025", "2024")
            if metadata.get("year"):
                year = metadata["year"]
                if year in query_lower:
                    score += 0.3
            
            # Check doc type match (e.g., "report", "transcript")
            if metadata.get("doc_type"):
                doc_type = metadata["doc_type"]
                if doc_type.lower() in query_lower:
                    score += 0.2
            
            # Check title match
            if metadata.get("title"):
                title = metadata["title"].lower()
                # Count matching words
                query_words = set(query_lower.split())
                title_words = set(title.split())
                overlap = len(query_words & title_words)
                if overlap > 0:
                    score += 0.1 * overlap
            
            if score > 0:
                metadata_scores[i] = score
        
        return metadata_scores
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.7, doc_boost: float = 0.3, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform two-stage hybrid search
        
        Stage 1: Search document metadata (title, quarter, year, doc_type)
        Stage 2: Hybrid semantic + BM25 search on content
        
        Args:
            query: User's question
            top_k: Number of results to return
            alpha: Weight for semantic search (0-1). 1.0 = pure semantic, 0.0 = pure keyword
            doc_boost: Boost multiplier for documents matching metadata (0-1)
            source_filter: If set, only return chunks from this source filename
        """
        if not self.documents:
            return []
        
        # Stage 1: Document metadata search
        metadata_scores = self._search_document_metadata(query)
        
        # Stage 2: Content search
        # 1. Semantic Search
        query_embedding = self.embedding_model.encode(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Map semantic scores to chunk index
        semantic_scores = {}
        for res in semantic_results:
            idx = self.documents.index(res['document'])
            semantic_scores[idx] = res['score']
            
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores (min-max normalization)
        if len(bm25_scores) > 0:
            min_score = min(bm25_scores)
            max_score = max(bm25_scores)
            if max_score > min_score:
                bm25_scores = (bm25_scores - min_score) / (max_score - min_score)
            else:
                bm25_scores = [0.0] * len(bm25_scores)
        
        # 3. Combine Scores with metadata boost
        hybrid_scores = []
        for i, doc in enumerate(self.documents):
            # Apply source filter: skip chunks not from the requested source
            if source_filter and doc.get("metadata", {}).get("source") != source_filter:
                continue
                
            sem_score = semantic_scores.get(i, 0.0)
            kw_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            meta_score = metadata_scores.get(i, 0.0)
            
            # Base hybrid score
            base_score = (alpha * sem_score) + ((1 - alpha) * kw_score)
            
            # Apply metadata boost
            final_score = base_score + (doc_boost * meta_score)
            
            hybrid_scores.append({
                "document": doc,
                "score": final_score,
                "semantic_score": sem_score,
                "bm25_score": kw_score,
                "metadata_score": meta_score
            })
            
        # Sort by final score
        hybrid_scores.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_scores[:top_k]
