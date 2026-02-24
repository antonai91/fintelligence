"""
DuckDB Agent for Table Queries

Ingests CSV tables into a local DuckDB database to allow advanced Text-to-SQL
analytical queries over financial data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import duckdb
from openai import OpenAI

from . import config

# Ensure Vector DB dir exists
config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

class DuckDBManager:
    """Manages syncing extracted CSV tables into DuckDB"""
    
    def __init__(self, db_path: Optional[str] = None, processed_dir: Optional[str] = None):
        self.db_path = str(db_path or config.TABLE_DB_PATH)
        self.processed_dir = Path(processed_dir or config.PROCESSED_DIR)
        
        # Connect to DB (creates if doesn't exist)
        self.con = duckdb.connect(self.db_path)
        
        # Create metadata catalog table to track ingested files and schemas
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS _table_catalog (
                csv_filename VARCHAR PRIMARY KEY,
                db_table_name VARCHAR,
                source_pdf VARCHAR,
                schema_json VARCHAR
            )
        """)

    def _sanitize_table_name(self, filename: str) -> str:
        """Convert a filename like 'Q1-2025-report_table_p4_1.csv' into a safe SQL table name"""
        name = filename.replace('.csv', '')
        # Replace non-alphanumeric chars with underscores
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        # Ensure it starts with a letter
        if not name[0].isalpha():
            name = 'tbl_' + name
        return name.lower()

    def sync_csvs(self):
        """Scan the processed directory and load any new CSVs into DuckDB"""
        csv_files = list(self.processed_dir.glob('**/*_table_*.csv'))
        
        # Get already ingested files
        res = self.con.execute("SELECT csv_filename FROM _table_catalog").fetchall()
        ingested = {row[0] for row in res}
        
        new_tables = 0
        for csv_path in csv_files:
            if csv_path.name in ingested:
                continue
                
            db_table_name = self._sanitize_table_name(csv_path.name)
            source_pdf = re.sub(r'_table_(?:p\d+_)?\d+\.csv$', '.pdf', csv_path.name)
            
            try:
                # Use DuckDB's automatic CSV reader to infer schema and create table
                self.con.execute(f"CREATE OR REPLACE TABLE {db_table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
                
                # Extract schema
                schema_res = self.con.execute(f"DESCRIBE {db_table_name}").fetchall()
                columns = [{"name": row[0], "type": row[1]} for row in schema_res]
                schema_json = json.dumps(columns)
                
                # Register in catalog
                self.con.execute(
                    "INSERT INTO _table_catalog VALUES (?, ?, ?, ?)", 
                    (csv_path.name, db_table_name, source_pdf, schema_json)
                )
                new_tables += 1
                
            except Exception as e:
                print(f"  ⚠ Failed to ingest {csv_path.name} into DuckDB: {e}")
                
        if new_tables > 0:
            print(f"📊 Registered {new_tables} new CSV tables in DuckDB.")

    def reload_csv(self, csv_filename: str):
        """Forcefully re-ingest a single CSV file, overwriting the existing table in DuckDB."""
        csv_path = self.processed_dir / csv_filename
        if not csv_path.exists():
            print(f"  ⚠ Cannot reload {csv_filename}, file not found.")
            return

        db_table_name = self._sanitize_table_name(csv_filename)
        source_pdf = re.sub(r'_table_(?:p\d+_)?\d+\.csv$', '.pdf', csv_filename)
        
        try:
            # Overwrite DuckDB table
            self.con.execute(f"CREATE OR REPLACE TABLE {db_table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
            
            # Extract schema
            schema_res = self.con.execute(f"DESCRIBE {db_table_name}").fetchall()
            columns = [{"name": row[0], "type": row[1]} for row in schema_res]
            schema_json = json.dumps(columns)
            
            # Overwrite catalog entry
            self.con.execute("DELETE FROM _table_catalog WHERE csv_filename = ?", (csv_filename,))
            self.con.execute(
                "INSERT INTO _table_catalog VALUES (?, ?, ?, ?)", 
                (csv_filename, db_table_name, source_pdf, schema_json)
            )
            print(f"✅ Reloaded {csv_filename} into DuckDB table {db_table_name}.")
        except Exception as e:
            print(f"  ⚠ Failed to reload {csv_filename} into DuckDB: {e}")

    def get_catalog_for_sources(self, source_pdfs: List[str]) -> List[Dict[str, Any]]:
        """Retrieve table schemas for specific source documents"""
        if not source_pdfs:
            return []
            
        placeholders = ', '.join(['?'] * len(source_pdfs))
        query = f"SELECT db_table_name, source_pdf, schema_json FROM _table_catalog WHERE source_pdf IN ({placeholders})"
        res = self.con.execute(query, source_pdfs).fetchall()
        
        catalog = []
        for row in res:
            catalog.append({
                "table_name": row[0],
                "source": row[1],
                "schema": json.loads(row[2])
            })
        return catalog
        
    def execute_query(self, query: str) -> str:
        """Execute a SQL query and return results as string (Markdown table if possible)"""
        try:
            # We fetch as a pandas dataframe to easily convert to markdown
            df = self.con.execute(query).df()
            if df.empty:
                return "Query executed successfully but returned no results."
            return df.to_markdown(index=False)
        except Exception as e:
            return f"Error executing query: {str(e)}"
            
    def close(self):
        self.con.close()


class TableQAAgent:
    """Agent that translates natural language queries into SQL for DuckDB"""
    
    def __init__(self, db_manager: DuckDBManager):
        self.db_manager = db_manager
        self._client = None
        
    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(api_key=config.get_openai_api_key())
        return self._client
        
    def query(self, question: str, source_pdfs: List[str]) -> Optional[str]:
        """
        Takes a user question and relevant document sources, generates SQL,
        executes it, and returns the result as string context.
        """
        catalog = self.db_manager.get_catalog_for_sources(source_pdfs)
        if not catalog:
            return None # No tables relevant for these sources
            
        # Format schema for the LLM
        schema_text = []
        for tbl in catalog:
            cols = [f"{c['name']} ({c['type']})" for c in tbl["schema"]]
            schema_text.append(f"Table Name: {tbl['table_name']}\nSource PDF: {tbl['source']}\nColumns: {', '.join(cols)}")
        
        schema_prompt = "\n\n".join(schema_text)
        
        system_prompt = """You are a Text-to-SQL agent for a financial DuckDB database. 
Given the user's question and the database schema, write a valid DuckDB SQL query to retrieve the answer.
Return ONLY a JSON object containing either:
1. "sql": the raw SQL query string. Do NOT use markdown code blocks (e.g. ```sql).
2. "error": if the question cannot be answered using the provided tables.

Always be careful with column names that have spaces or special characters (quote them).
"""

        user_prompt = f"""Available Database Tables:
{schema_prompt}

User Question: {question}

JSON response:"""

        try:
            print("  🤔 SQL Agent is reasoning about table structures...")
            response = self.client.chat.completions.create(
                model=config.MODEL_QA,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            # Clean markdown if present
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                
            data = json.loads(result)
            
            if "sql" in data:
                sql_query = data["sql"]
                print(f"  ⚡ Executing SQL: {sql_query}")
                query_result = self.db_manager.execute_query(sql_query)
                return f"Analytics Result from Database:\n{query_result}"
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠ SQL Agent failed: {e}")
            return None
