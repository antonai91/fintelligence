import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import duckdb

from investor_relations_scraper.table_db import DuckDBManager


def test_reload_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        processed_dir = Path(tmpdir) / "processed"
        processed_dir.mkdir()
        
        manager = DuckDBManager(db_path=str(db_path), processed_dir=str(processed_dir))
        
        # 1. Create initial CSV
        csv_filename = "test_doc_table_p1_1.csv"
        csv_path = processed_dir / csv_filename
        
        df_initial = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        df_initial.to_csv(csv_path, index=False)
        
        # Sync to ingest
        manager.sync_csvs()
        
        # Verify initial data
        res = manager.con.execute("SELECT * FROM test_doc_table_p1_1").df()
        assert len(res) == 2
        assert res.iloc[0]["col1"] == 1
        
        # 2. Overwrite CSV with new data
        df_updated = pd.DataFrame({"col1": [99, 100, 101], "col2": ["X", "Y", "Z"]})
        df_updated.to_csv(csv_path, index=False)
        
        # 3. Reload specifically
        manager.reload_csv(csv_filename)
        
        # 4. Verify updated data
        res_updated = manager.con.execute("SELECT * FROM test_doc_table_p1_1").df()
        assert len(res_updated) == 3
        assert res_updated.iloc[0]["col1"] == 99
        
        # 5. Verify catalog
        catalog_res = manager.con.execute("SELECT * FROM _table_catalog WHERE csv_filename = ?", (csv_filename,)).fetchall()
        assert len(catalog_res) == 1
        
        manager.close()
        print("✅ DuckDB reload testing passed successfully!")

if __name__ == "__main__":
    test_reload_csv()
