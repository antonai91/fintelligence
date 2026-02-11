
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to sys.path so we can import the package
sys.path.append(str(Path(__file__).parents[1] / "src"))

from investor_relations_scraper import EquinorScraper

async def run_scraper_examples():
    print("="*50)
    print("Equinor Scraper - Programmatic Usage Examples")
    print("="*50)

    # Example 1: Download only 2025 reports to a specific folder
    print("\n\n--- Example 1: Downloading 2025 Reports ---")
    custom_dir = "data/example_2025"
    print(f"Goal: Download 2025 reports to '{custom_dir}'")
    
    scraper_2025 = EquinorScraper(download_dir=custom_dir)
    await scraper_2025.scrape(headless=True, year_filter="2025")
    
    # Check what was downloaded
    if os.path.exists(custom_dir):
        files = os.listdir(custom_dir)
        print(f"\nFiles downloaded to {custom_dir}:")
        for f in files:
            print(f" - {f}")
    
    # Example 2: Download only 2024 reports
    print("\n\n--- Example 2: Downloading 2024 Reports ---")
    custom_dir_2024 = "data/example_2024"
    print(f"Goal: Download 2024 reports to '{custom_dir_2024}'")
    
    scraper_2024 = EquinorScraper(download_dir=custom_dir_2024)
    await scraper_2024.scrape(headless=True, year_filter="2024")
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(run_scraper_examples())
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
