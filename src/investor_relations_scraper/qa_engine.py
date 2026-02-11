import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import hashlib
import faiss

from . import config

class ProcessedDocumentLoader:
    """Loads processed text files and chunks them with rich metadata"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._openai_client = None
        self._metadata_cache = {}  # Cache to avoid re-extracting for same files
        
    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client for metadata extraction"""
        if self._openai_client is None:
            api_key = config.get_openai_api_key()
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client
        
    def _extract_metadata_with_llm(self, content: str, filename: str) -> Dict[str, str]:
        """
        Extract metadata from document content using LLM (GPT-4)
        
        This analyzes the actual content of the document to extract:
        - Quarter (Q1, Q2, Q3, Q4)
        - Year (2024, 2025, etc.)
        - Document type (report, transcript, presentation, financial-statements)
        - Company name
        - A descriptive title
        """
        # Use first ~3000 chars for metadata extraction (usually contains title, date, etc.)
        sample_content = content[:3000] if len(content) > 3000 else content
        
        # Remove _text.txt suffix for base filename
        base_name = filename.replace('_text.txt', '').replace('.txt', '')
        
        prompt = f"""Analyze this financial document and extract the following metadata.
Return ONLY a JSON object with these fields (no markdown, no explanation):

{{
    "quarter": "Q1" or "Q2" or "Q3" or "Q4" or null (if not quarterly),
    "year": "2024" (4-digit year string) or null,
    "doc_type": one of ["report", "transcript", "presentation", "financial-statements", "annual-report", "other"],
    "company": "Company Name" or null,
    "title": "A short descriptive title for this document"
}}

Filename: {base_name}

Document content (first portion):
{sample_content}

JSON:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.MODEL_METADATA,
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant. Extract structured metadata from financial documents. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            
            import json
            metadata = json.loads(result)
            
            # Ensure all expected fields exist
            return {
                "title": metadata.get("title") or base_name,
                "quarter": metadata.get("quarter"),
                "year": metadata.get("year"),
                "doc_type": metadata.get("doc_type") or "unknown",
                "company": metadata.get("company")
            }
            
        except Exception as e:
            print(f"  Warning: LLM metadata extraction failed for {filename}: {e}")
            # Fallback to filename-based extraction
            return self._extract_metadata_from_filename(filename)
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Fallback: Extract metadata from filename using regex patterns
        
        Examples:
            Q1-2025-report_text.txt -> {quarter: Q1, year: 2025, doc_type: report}
        """
        # Remove _text.txt suffix
        base_name = filename.replace('_text.txt', '').replace('.txt', '')
        
        metadata = {
            "title": base_name,
            "quarter": None,
            "year": None,
            "doc_type": "unknown",
            "company": None
        }
        
        # Extract quarter (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'Q([1-4])', base_name, re.IGNORECASE)
        if quarter_match:
            metadata["quarter"] = f"Q{quarter_match.group(1)}"
        
        # Extract year (2024, 2025, etc.)
        year_match = re.search(r'(20\d{2})', base_name)
        if year_match:
            metadata["year"] = year_match.group(1)
        
        # Determine document type
        lower_name = base_name.lower()
        if 'transcript' in lower_name:
            metadata["doc_type"] = "transcript"
        elif 'presentation' in lower_name:
            metadata["doc_type"] = "presentation"
        elif 'report' in lower_name:
            metadata["doc_type"] = "report"
        elif 'financial' in lower_name or 'statement' in lower_name:
            metadata["doc_type"] = "financial-statements"
        
        return metadata
        
    def extract_text_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load processed text files from directory
        
        Returns:
            List of documents (chunks) with rich metadata
        """
        documents = []
        path = Path(directory)
        
        # Look for processed text files
        text_files = list(path.glob('**/*_text.txt'))
        if not text_files:
            # Fallback to any .txt files
            text_files = list(path.glob('**/*.txt'))
        
        print(f"Found {len(text_files)} processed text files in {directory}")
        
        for file_path in text_files:
            try:
                print(f"Loading {file_path.name}...")
                chunks = self._process_file(file_path)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return documents
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single text file into chunks with metadata"""
        # Read the processed text
        text = file_path.read_text(encoding='utf-8')
        
        # Extract metadata using LLM (analyzes content, not just filename)
        print(f"  Extracting metadata with LLM...")
        file_metadata = self._extract_metadata_with_llm(text, file_path.name)
        print(f"  -> {file_metadata.get('title')} | {file_metadata.get('quarter')} {file_metadata.get('year')} | {file_metadata.get('doc_type')}")
        
        chunks = []
        
        # Simple word-based chunking
        words = text.split()
        
        chunk_id = 0
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Skip very small chunks
            if len(chunk_words) < 50:
                continue
            
            # Create chunk with rich metadata (including company from LLM)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": file_path.name.replace('_text.txt', '.pdf'),  # Original PDF name
                    "title": file_metadata["title"],
                    "doc_type": file_metadata["doc_type"],
                    "quarter": file_metadata["quarter"],
                    "year": file_metadata["year"],
                    "company": file_metadata.get("company"),
                    "path": str(file_path),
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
            
        return chunks
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


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
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.7, doc_boost: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform two-stage hybrid search
        
        Stage 1: Search document metadata (title, quarter, year, doc_type)
        Stage 2: Hybrid semantic + BM25 search on content
        
        Args:
            query: User's question
            top_k: Number of results to return
            alpha: Weight for semantic search (0-1). 1.0 = pure semantic, 0.0 = pure keyword
            doc_boost: Boost multiplier for documents matching metadata (0-1)
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


class QAEngine:
    """Main interface for Question Answering"""
    
    def __init__(self, data_dir: str, persist_directory: Optional[str] = None, 
                 enable_memory: bool = True, max_history: int = 10):
        self.data_dir = data_dir
        self.processor = ProcessedDocumentLoader()  # Changed from PDFProcessor
        persist_directory = persist_directory or str(config.VECTOR_DB_DIR)
        self.search_engine = HybridSearchEngine(persist_directory=persist_directory)
        self._client = None  # Lazy initialization
        self.is_indexed = False
        
        # Conversation memory
        self.enable_memory = enable_memory
        if enable_memory:
            memory_path = Path(persist_directory) / "conversation_history.pkl"
            self.memory = ConversationMemory(max_messages=max_history, persist_path=str(memory_path))
            self.memory.load()  # Try to load existing conversation
        else:
            self.memory = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            api_key = config.get_openai_api_key()
            self._client = OpenAI(api_key=api_key)
        return self._client
        
    def _get_current_file_hashes(self) -> Dict[str, str]:
        """Get hashes of all text files in data directory"""
        file_hashes = {}
        path = Path(self.data_dir)
        text_files = list(path.glob('**/*_text.txt'))
        if not text_files:
            text_files = list(path.glob('**/*.txt'))
        
        for file_path in text_files:
            file_hash = ProcessedDocumentLoader.get_file_hash(str(file_path))
            file_hashes[str(file_path)] = file_hash
            
        return file_hashes
        
    def _check_if_reindex_needed(self) -> Tuple[bool, List[str]]:
        """
        Check if re-indexing is needed by comparing file hashes
        
        Returns:
            (needs_reindex, changed_files)
        """
        current_hashes = self._get_current_file_hashes()
        stored_hashes = self.search_engine.vector_store.file_hashes
        
        # Check for new or modified files
        changed_files = []
        for file_path, current_hash in current_hashes.items():
            if file_path not in stored_hashes or stored_hashes[file_path] != current_hash:
                changed_files.append(file_path)
        
        # Check for deleted files
        for file_path in stored_hashes:
            if file_path not in current_hashes:
                changed_files.append(file_path)
        
        return len(changed_files) > 0, changed_files
        
    def load_and_index(self, force_reindex: bool = False):
        """Load PDFs and build the index (with smart caching)"""
        
        # Try to load existing index
        if not force_reindex and self.search_engine.load_existing_index():
            print("\n✓ Loaded existing index from disk")
            
            # Check if files have changed
            needs_reindex, changed_files = self._check_if_reindex_needed()
            
            if not needs_reindex:
                print("✓ All files are up to date, no re-indexing needed!")
                self.is_indexed = True
                return
            else:
                print(f"\n⚠ Detected changes in {len(changed_files)} file(s):")
                for f in changed_files[:5]:  # Show first 5
                    print(f"  - {Path(f).name}")
                if len(changed_files) > 5:
                    print(f"  ... and {len(changed_files) - 5} more")
                print("\nRe-indexing all documents...")
        
        # Full indexing
        print("Loading documents...")
        documents = self.processor.extract_text_from_directory(self.data_dir)
        
        if not documents:
            print("No documents found to index.")
            return
        
        # Get current file hashes
        file_hashes = self._get_current_file_hashes()
            
        print(f"Indexing {len(documents)} text chunks...")
        self.search_engine.index_documents(documents, file_hashes)
        self.is_indexed = True
        print("✓ Indexing complete!")
        
    def answer_question(self, question: str, use_memory: Optional[bool] = None) -> Dict[str, Any]:
        """
        Answer a question using the indexed documents
        
        Args:
            question: The user's question
            use_memory: Override the default memory setting for this question
        """
        if not self.is_indexed:
            self.load_and_index()
        
        # Determine if we should use memory for this question
        use_mem = use_memory if use_memory is not None else self.enable_memory
            
        # Retrieval
        print(f"Searching for context for: '{question}'...")
        results = self.search_engine.search(question, top_k=5)
        
        if not results:
            answer = "I couldn't find any relevant information in the documents."
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", answer)
                self.memory.save()
            return {
                "answer": answer,
                "sources": []
            }
            
        # Context Construction
        context_text = ""
        sources = []
        
        for i, res in enumerate(results):
            doc = res["document"]
            source_name = doc["metadata"]["source"]
            context_text += f"Source {i+1} ({source_name}):\n{doc['text']}\n\n"
            
            if source_name not in sources:
                sources.append(source_name)
                
        # LLM Generation with conversation history
        system_prompt = """You are a helpful financial analyst assistant. 
Use the provided context to answer the user's question about Equinor. 
If the answer is not in the context, say you don't know. 
Cite the sources (Source 1, Source 2, etc.) when stating facts.
Keep the answer concise and professional.
If there is previous conversation history, use it to provide more contextual and relevant answers."""

        # Build the user prompt with optional conversation history
        user_prompt = ""
        
        # Add conversation history if memory is enabled
        if use_mem and self.memory and self.memory.messages:
            user_prompt += self.memory.get_formatted_history()
            user_prompt += "---\n\n"
        
        user_prompt += f"""Context from documents:
{context_text}

Current question: {question}

Answer:"""

        print("Generating answer with OpenAI...")
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_QA,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_RESPONSE_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Add to conversation memory
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", answer)
                self.memory.save()
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": results
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", error_msg)
                self.memory.save()
            return {
                "answer": error_msg,
                "sources": sources
            }
    
    def clear_conversation(self):
        """Clear the conversation history"""
        if self.memory:
            self.memory.clear()
            self.memory.save()
            print("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        if self.memory:
            return self.memory.get_history()
        return []

