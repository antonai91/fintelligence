"""
QA Engine — Agentic Plan → Retrieve → Synthesize Pipeline

Main interface for question answering over indexed financial documents.
Uses OpenAI for planning and synthesis, with hybrid search for retrieval.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from . import config
from .document_loader import ProcessedDocumentLoader
from .search import PersistentVectorStore, HybridSearchEngine
from .conversation_memory import ConversationMemory
from .table_db import DuckDBManager, TableQAAgent



def _build_source_refs(results: list) -> list:
    """
    Deduplicate retrieved chunks into a list of {pdf, page, title} dicts,
    ordered by document then page number, for the source navigation panel.
    """
    seen = set()
    refs = []
    for res in results:
        meta = res.get("document", {}).get("metadata", {})
        source = meta.get("source", "")
        page = meta.get("page")
        title = meta.get("title", source)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            refs.append({"pdf": source, "page": page, "title": title})
    # Sort by source then page
    refs.sort(key=lambda r: (r["pdf"], r["page"] or 0))
    return refs


class QAEngine:
    """Main interface for Question Answering"""
    
    def __init__(self, data_dir: str, persist_directory: Optional[str] = None, 
                 enable_memory: bool = True, max_history: int = 10):
        self.data_dir = data_dir
        self.processor = ProcessedDocumentLoader()  # Changed from PDFProcessor
        persist_directory = persist_directory or str(config.VECTOR_DB_DIR)
        self.search_engine = HybridSearchEngine(persist_directory=persist_directory)
        self.db_manager = DuckDBManager(processed_dir=data_dir)
        self.sql_agent = TableQAAgent(self.db_manager)
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
        
        # Sync CSV tables into DuckDB
        print("Syncing CSV tables into DuckDB database...")
        self.db_manager.sync_csvs()
        
        self.is_indexed = True
        print("✓ Indexing complete!")
        
    def _get_document_catalog(self) -> Tuple[List[Dict[str, str]], str]:
        """
        Build a deduplicated catalog of available documents with metadata.
        
        Returns:
            Tuple of (catalog list, formatted catalog string for LLM)
        """
        seen_sources = {}
        table_summaries = {}  # Collect table summaries per source
        
        for doc in self.search_engine.documents:
            meta = doc.get("metadata", {})
            source = meta.get("source", "unknown")
            doc_type = meta.get("doc_type", "unknown")
            
            if source not in seen_sources:
                seen_sources[source] = {
                    "source": source,
                    "title": meta.get("title", source),
                    "quarter": meta.get("quarter"),
                    "year": meta.get("year"),
                    "doc_type": doc_type,
                    "company": meta.get("company")
                }
                
        # Fetch tables ingested into DuckDB to annotate catalog with has_tables flag
        with self.db_manager._connect() as con:
            res_tables = con.execute("SELECT DISTINCT source_pdf FROM _table_catalog").fetchall()
        for row in res_tables:
            src = row[0]
            if src in seen_sources:
                seen_sources[src]["has_tables"] = True
        
        catalog = list(seen_sources.values())
        
        # Format for the LLM
        lines = []
        for i, entry in enumerate(catalog, 1):
            q = entry['quarter'] or 'N/A'
            y = entry['year'] or 'N/A'
            line = f"{i}. {entry['source']} — {entry['title']} | {q} {y} | Type: {entry['doc_type']}"
            
            # Indicate tables are available
            if entry.get("has_tables", False):
                line += f"\n   📊 Contains queryable tabular data in SQL Database."
            
            lines.append(line)
        
        catalog_str = "\n".join(lines)
        return catalog, catalog_str
    
    def _plan_sources(self, question: str, catalog_str: str, conversation_context: str = "") -> Dict[str, Any]:
        """
        Use the LLM to reason about which sources are needed.
        
        Args:
            question: The user's question
            catalog_str: Formatted document catalog
            conversation_context: Optional conversation history
            
        Returns:
            Dict with 'reasoning' and 'sources' list
        """
        prompt = f"""You are a financial research planner. Given a user question and a catalog of available documents, 
your job is to think step-by-step about which documents are needed to answer the question comprehensively.

{conversation_context}Available documents:
{catalog_str}

User question: {question}

Think step-by-step:
1. What is the user really asking? (Identify the time period, topic, scope)
2. Which documents cover the relevant time period(s)?
3. Which document types (report, transcript, presentation, table) are most useful for this question?
4. For questions about specific numbers, figures, or financial metrics, prefer sources that have associated tables (marked with 📊)
5. Are there any edge cases? (e.g., "2025" means ALL quarters of 2025, "last year" depends on context)

Then return a JSON object with:
- "reasoning": A brief explanation of your thinking (2-3 sentences)
- "sources": A list of objects, each with "source" (exact filename from the catalog) and "chunks" (number of chunks to retrieve, 1-5)

Rules:
- Select ALL documents that are relevant, don't leave any out
- For broad time-based questions (e.g., "what happened in 2025"), include ALL quarters
- For comparison questions, include all documents being compared
- For quantitative questions (revenue, costs, dividends, production), prioritize sources with tables
- Maximum {config.MAX_PLANNED_SOURCES} sources
- Default to {config.CHUNKS_PER_SOURCE_DEFAULT} chunks per source unless you have a reason to adjust

Return ONLY the JSON object, no markdown formatting:"""

        try:
            print("  🧠 Planning: deciding which sources to use...")
            response = self.client.chat.completions.create(
                model=config.MODEL_QA,
                messages=[
                    {"role": "system", "content": "You are a research planning assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean markdown if present
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            
            plan = json.loads(result)
            
            # Validate structure
            if "sources" not in plan or not isinstance(plan["sources"], list):
                raise ValueError("Invalid plan structure: missing 'sources' list")
            
            # Cap sources
            plan["sources"] = plan["sources"][:config.MAX_PLANNED_SOURCES]
            
            reasoning = plan.get("reasoning", "No reasoning provided")
            print(f"  💡 Reasoning: {reasoning}")
            print(f"  📋 Selected {len(plan['sources'])} source(s): {[s['source'] for s in plan['sources']]}")
            
            return plan
            
        except Exception as e:
            print(f"  ⚠ Planning failed ({e}), falling back to all sources")
            # Fallback: return all sources with default chunks
            return {
                "reasoning": f"Planning failed ({e}), using all available sources",
                "sources": [{"source": s, "chunks": config.CHUNKS_PER_SOURCE_DEFAULT} 
                           for s in set(d.get("metadata", {}).get("source", "") 
                                       for d in self.search_engine.documents)]
            }
    
    def _retrieve_for_plan(self, question: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from the sources specified in the plan.
        
        Args:
            question: The user's question  
            plan: The source plan from _plan_sources
            
        Returns:
            List of search results across all planned sources
        """
        all_results = []
        
        for source_plan in plan.get("sources", []):
            source_name = source_plan.get("source", "")
            chunks_to_get = source_plan.get("chunks", config.CHUNKS_PER_SOURCE_DEFAULT)
            
            # Search within this specific source
            results = self.search_engine.search(
                query=question,
                top_k=chunks_to_get,
                source_filter=source_name
            )
            
            if results:
                all_results.extend(results)
            else:
                print(f"  ⚠ No chunks found for source: {source_name}")
        
        # Sort all results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results
    
    def answer_question(self, question: str, use_memory: Optional[bool] = None) -> Dict[str, Any]:
        """
        Answer a question using the agentic Plan → Retrieve → Synthesize pipeline.
        
        Args:
            question: The user's question
            use_memory: Override the default memory setting for this question
        """
        if not self.is_indexed:
            self.load_and_index()
        
        # Determine if we should use memory for this question
        use_mem = use_memory if use_memory is not None else self.enable_memory
        
        # === Stage 1: PLAN ===
        print(f"\n🔍 Processing question: '{question}'")
        
        catalog, catalog_str = self._get_document_catalog()
        
        if not catalog:
            answer = "I couldn't find any documents in the index."
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", answer)
                self.memory.save()
            return {"answer": answer, "sources": [], "plan": None}
        
        # Build conversation context for the planner
        conversation_context = ""
        if use_mem and self.memory and self.memory.messages:
            conversation_context = self.memory.get_formatted_history() + "---\n\n"
        
        plan = self._plan_sources(question, catalog_str, conversation_context)
        
        # === Stage 2: RETRIEVE ===
        print("  📚 Retrieving relevant chunks from selected sources...")
        results = self._retrieve_for_plan(question, plan)
        
        if not results:
            answer = "I couldn't find any relevant information in the selected documents."
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", answer)
                self.memory.save()
            return {
                "answer": answer,
                "sources": [],
                "plan": plan
            }
        
        # === Stage 3: SYNTHESIZE ===
        # Build context grouped by source
        context_text = ""
        sources = []
        
        for i, res in enumerate(results):
            doc = res["document"]
            source_name = doc["metadata"]["source"]
            quarter = doc["metadata"].get("quarter", "")
            year = doc["metadata"].get("year", "")
            # Guard against list values from LLM metadata extraction
            if isinstance(quarter, list):
                quarter = quarter[0] if quarter else ""
            if isinstance(year, list):
                year = year[0] if year else ""
            context_text += f"Source {i+1} ({source_name} — {quarter} {year}):\n{doc['text']}\n\n"
            
            if source_name not in sources:
                sources.append(source_name)
        
        reasoning = plan.get("reasoning", "")
        
        system_prompt = """You are a helpful financial analyst assistant.
Use the provided context to answer the user's question about Equinor.
If the answer is not in the context, say you don't know.
Cite the sources (Source 1, Source 2, etc.) when stating facts.
Provide a comprehensive, well-structured answer. Use bullet points or sections for clarity when appropriate.
If the context contains a SQL Analytics Result from DuckDB, heavily prioritize that analytical data to form your exact numerical answer.
If there is previous conversation history, use it to provide more contextual and relevant answers."""

        user_prompt = ""
        
        if use_mem and self.memory and self.memory.messages:
            user_prompt += self.memory.get_formatted_history()
            user_prompt += "---\n\n"
            
        # Optional: Query DuckDB Text-to-SQL logic if tables are available for selected sources
        source_names = [s['source'] for s in plan.get("sources", [])]
        sql_context = self.sql_agent.query(question, source_names)
        
        if sql_context:
            print("  ✅ Received analytical result from DuckDB Agent to fuse into synthesis!")
            context_text = f"**{sql_context}**\n\n---\n\n" + context_text
        
        user_prompt += f"""Research plan reasoning: {reasoning}

Context from {len(sources)} document(s):
{context_text}

Question: {question}

Answer:"""

        print(f"  🤖 Synthesizing answer from {len(sources)} source(s), {len(results)} chunks...")
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
            
            if use_mem and self.memory:
                self.memory.add_message("user", question)
                self.memory.add_message("assistant", answer)
                self.memory.save()
            
            return {
                "answer": answer,
                "sources": sources,
                "source_refs": _build_source_refs(results),
                "plan": plan,
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
                "sources": sources,
                "source_refs": [],
                "plan": plan
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
