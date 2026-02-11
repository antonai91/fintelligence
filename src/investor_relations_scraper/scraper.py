"""
Equinor Investor Relations PDF Scraper

This script uses Playwright to navigate to Equinor's quarterly results page
and download all available financial report PDFs.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict
from playwright.async_api import async_playwright, Page, Download
import re

download_folder = "data/raw/"

class EquinorScraper:
    """Scraper for downloading Equinor investor relations PDFs"""
    
    def __init__(self, download_dir: str = download_folder):
        """
        Initialize the scraper
        
        Args:
            download_dir: Directory where PDFs will be downloaded
        """
        self.base_url = "https://www.equinor.com/investors/quarterly-results"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    async def wait_for_page_load(self, page: Page):
        """Wait for the page to fully load"""
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(1)  # Additional wait for dynamic content
        
    async def scroll_to_bottom(self, page: Page):
        """Scroll to the bottom of the page to load all content"""
        await page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        
                        if(totalHeight >= scrollHeight){
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)
        
    async def extract_pdf_links(self, page: Page) -> List[Dict[str, str]]:
        """
        Extract all PDF download links from the page by parsing __NEXT_DATA__
        
        Returns:
            List of dictionaries containing title and URL for each PDF
        """
        # Wait for content to load
        await self.wait_for_page_load(page)
        await page.wait_for_timeout(2000)
        
        # Extract PDF information from __NEXT_DATA__ script tag (Next.js data)
        print("Extracting PDFs from page data...")
        pdf_data = await page.evaluate("""
            () => {
                try {
                    // Parse the Next.js data
                    const nextDataEl = document.getElementById('__NEXT_DATA__');
                    if (!nextDataEl) return [];
                    
                    const nextData = JSON.parse(nextDataEl.innerHTML);
                    const content = nextData?.props?.pageProps?.data?.pageData?.content;
                    if (!content) return [];
                    
                    const reports = [];
                    
                    // Iterate through content blocks to find downloadable files
                    content.forEach(block => {
                        if (block.callToActions) {
                            block.callToActions.forEach(action => {
                                if (action.type === 'downloadableFile' && action.file) {
                                    reports.push({
                                        title: action.label ? action.label.trim() : 'Report',
                                        url: action.file.url,
                                        filename: action.file.originalFilename || null
                                    });
                                }
                            });
                        }
                    });
                    
                    return reports;
                } catch (error) {
                    console.error('Error extracting PDFs:', error);
                    return [];
                }
            }
        """)
        
        return pdf_data
    
    async def download_pdf(self, url: str, filename: str) -> bool:
        """
        Download a PDF file using direct HTTP download
        
        Args:
            url: URL of the PDF
            filename: Filename to save as
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            # Create a clean filename
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = re.sub(r'[-\s]+', '-', filename).strip('-')
            
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = self.download_dir / filename
            
            # Ensure download directory exists
            self.download_dir.mkdir(parents=True, exist_ok=True)
            
            # Use direct download method
            return await self._download_direct(url, filepath, filename)
            
        except Exception as e:
            print(f"✗ Failed to download {filename}: {str(e)}")
            return False
    
    async def _download_direct(self, url: str, filepath: Path, filename: str) -> bool:
        """
        Direct download method using aiohttp
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    print(f"✓ Downloaded (direct method): {filepath.name}")
                    return True
                else:
                    print(f"✗ HTTP {response.status} for {url}")
                    return False
    
    async def scrape(self, headless: bool = True, year_filter: str = None):
        """
        Main scraping method
        
        Args:
            headless: Run browser in headless mode
            year_filter: Optional year to filter reports (e.g., "2025", "2024")
        """
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            print(f"Navigating to {self.base_url}...")
            await page.goto(self.base_url)
            await self.wait_for_page_load(page)
            
            print("Extracting PDF links...")
            pdf_links = await self.extract_pdf_links(page)
            
            if not pdf_links:
                print("⚠ No PDF links found. The page structure may have changed.")
                await browser.close()
                return
            
            # Filter by year if specified
            if year_filter:
                pdf_links = [
                    pdf for pdf in pdf_links 
                    if year_filter in pdf['title'] or year_filter in pdf['url']
                ]
                print(f"Filtered to {year_filter}: {len(pdf_links)} reports found")
            
            print(f"Found {len(pdf_links)} PDF reports to download")
            
            # Download each PDF
            for i, pdf_info in enumerate(pdf_links, 1):
                title = pdf_info['title']
                url = pdf_info['url']
                
                # Create filename from title or URL
                if title and title != 'Unknown':
                    filename = title
                else:
                    # Extract filename from URL
                    filename = url.split('/')[-1].split('?')[0]
                
                print(f"\n[{i}/{len(pdf_links)}] Downloading: {title}")
                await self.download_pdf(url, filename)
            
            await browser.close()
            print(f"\n✓ Scraping complete! Files saved to: {self.download_dir.absolute()}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Equinor investor relations PDFs')
    parser.add_argument('--dir', default=download_folder, help=f'Download directory (default: {download_folder})')
    parser.add_argument('--year', help='Filter by year (e.g., 2025, 2024)')
    parser.add_argument('--no-headless', action='store_true', help='Show browser window')
    
    args = parser.parse_args()
    
    scraper = EquinorScraper(download_dir=args.dir)
    await scraper.scrape(headless=not args.no_headless, year_filter=args.year)


if __name__ == "__main__":
    asyncio.run(main())