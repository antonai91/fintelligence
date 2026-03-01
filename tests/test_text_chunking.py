import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from investor_relations_scraper.cli import PDFExtractor
from investor_relations_scraper import config

class TestTextChunking:
    @pytest.fixture
    def extractor(self):
        """Create a PDFExtractor instance with mocked dependencies."""
        with patch('investor_relations_scraper.cli.AsyncOpenAI'), \
             patch('investor_relations_scraper.config.get_openai_api_key', return_value="fake-key"):
            extractor = PDFExtractor(api_key="fake-key")
            # Mock the client
            extractor.client = AsyncMock()
            return extractor

    def test_chunk_text_small(self, extractor):
        """Test that small text is not chunked."""
        text = "This is a small text."
        chunks = extractor._chunk_text(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_paragraphs(self, extractor):
        """Test chunking by paragraphs."""
        para1 = "a" * 40
        para2 = "b" * 40
        para3 = "c" * 40
        text = f"{para1}\n\n{para2}\n\n{para3}"
        
        # Max chars enough for one paragraph but not two
        chunks = extractor._chunk_text(text, max_chars=50)
        
        assert len(chunks) == 3
        assert chunks[0] == para1
        assert chunks[1] == para2
        assert chunks[2] == para3

    def test_chunk_text_lines(self, extractor):
        """Test chunking by lines when a paragraph is too long."""
        line1 = "a" * 30
        line2 = "b" * 30
        text = f"{line1}\n{line2}"
        
        # Max chars enough for one line but not two
        chunks = extractor._chunk_text(text, max_chars=40)
        
        assert len(chunks) == 2
        assert chunks[0] == line1
        assert chunks[1] == line2

    def test_chunk_text_hard_split(self, extractor):
        """Test hard splitting when a line is too long."""
        text = "a" * 100
        
        chunks = extractor._chunk_text(text, max_chars=30)
        
        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10

    @pytest.mark.asyncio
    async def test_clean_text_with_openai_no_chunking(self, extractor):
        """Test cleaning text that doesn't need chunking."""
        text = "Small text"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Cleaned small text"
        extractor.client.chat.completions.create.return_value = mock_response

        result = await extractor.clean_text_with_openai(text, "test.pdf")
        
        assert result == "Cleaned small text"
        extractor.client.chat.completions.create.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_clean_text_with_openai_chunking(self, extractor):
        """Test cleaning text that requires chunking."""
        # Create text larger than MAX_TEXT_CHARS
        # We'll mock MAX_TEXT_CHARS for this test to be small
        with patch('investor_relations_scraper.config.MAX_TEXT_CHARS', 15):
            text = "Part 1 text.\n\nPart 2 text."
            
            # Setup mock to return different cleaned text for each call
            mock_response1 = MagicMock()
            mock_response1.choices[0].message.content = "Cleaned Part 1"
            
            mock_response2 = MagicMock()
            mock_response2.choices[0].message.content = "Cleaned Part 2"
            
            extractor.client.chat.completions.create.side_effect = [mock_response1, mock_response2]

            result = await extractor.clean_text_with_openai(text, "test.pdf")
            
            assert "Cleaned Part 1" in result
            assert "Cleaned Part 2" in result
            assert "\n\n" in result
            assert extractor.client.chat.completions.create.call_count == 2
