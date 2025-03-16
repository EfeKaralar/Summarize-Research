import os
import requests
import json
from typing import List, Dict, Optional, Union
import arxiv
import PyPDF2
from io import BytesIO
import re
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class to store paper information."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    published_date: Optional[str] = None
    full_text: Optional[str] = None
    summary: Optional[str] = None

class PaperFetcher:
    """Base class for fetching papers from different sources."""
    def fetch_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Fetch papers based on a query."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def fetch_paper_content(self, paper: Paper) -> Paper:
        """Fetch the full text content of a paper."""
        raise NotImplementedError("Subclasses must implement this method")

class ArxivFetcher(PaperFetcher):
    """Fetcher for papers from arXiv."""
    def fetch_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Fetch papers from arXiv based on a query."""
        logger.info(f"Fetching papers from arXiv with query: {query}")
        
        # Handle both author and field queries
        if query.startswith("author:"):
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
        else:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
        
        papers = []
        for result in search.results():
            authors = [author.name for author in result.authors]
            paper = Paper(
                title=result.title,
                authors=authors,
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                published_date=result.published.strftime("%Y-%m-%d") if result.published else None
            )
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers on arXiv")
        return papers
    
    def fetch_paper_content(self, paper: Paper) -> Paper:
        """Fetch the full text content of a paper from arXiv."""
        if not paper.pdf_url:
            logger.warning(f"No PDF URL available for paper: {paper.title}")
            return paper
        
        try:
            logger.info(f"Downloading PDF from {paper.pdf_url}")
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            
            # Read PDF content
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            
            # Clean up the text
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            paper.full_text = text
            return paper
            
        except Exception as e:
            logger.error(f"Error fetching PDF content: {str(e)}")
            return paper

class LLMSummarizer:
    """Base class for LLM-based summarizers."""
    def summarize(self, paper: Paper) -> str:
        """Summarize a paper using an LLM."""
        raise NotImplementedError("Subclasses must implement this method")

class AnthropicSummarizer(LLMSummarizer):
    """Summarizer using Anthropic's Claude API."""
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    def _create_prompt(self, paper: Paper) -> str:
        """Create a prompt for the LLM."""
        prompt = f"""
        Please provide a comprehensive summary of the following research paper. Structure your summary with these sections:
        
        1. Key Findings: The main discoveries or contributions in 3-5 bullet points
        2. Research Question/Problem: What problem is the paper trying to solve?
        3. Methodology: How did they approach the problem?
        4. Results: What were the outcomes?
        5. Implications: Why these findings matter for the field
        
        Here's the paper information:
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Published: {paper.published_date or 'Not specified'}
        
        Abstract:
        {paper.abstract}
        
        Full Text (excerpt):
        {paper.full_text[:10000] if paper.full_text else 'Not available'}
        """
        return prompt
    
    def summarize(self, paper: Paper) -> str:
        """Summarize a paper using Anthropic's Claude API."""
        prompt = self._create_prompt(paper)
        
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result["content"][0]["text"]
            return summary
        except Exception as e:
            logger.error(f"Error using Anthropic API: {str(e)}")
            return f"Error generating summary: {str(e)}"

class OpenAISummarizer(LLMSummarizer):
    """Summarizer using OpenAI's API."""
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _create_prompt(self, paper: Paper) -> str:
        """Create a prompt for the LLM."""
        prompt = f"""
        Please provide a comprehensive summary of the following research paper. Structure your summary with these sections:
        
        1. Key Findings: The main discoveries or contributions in 3-5 bullet points
        2. Research Question/Problem: What problem is the paper trying to solve?
        3. Methodology: How did they approach the problem?
        4. Results: What were the outcomes?
        5. Implications: Why these findings matter for the field
        
        Here's the paper information:
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Published: {paper.published_date or 'Not specified'}
        
        Abstract:
        {paper.abstract}
        
        Full Text (excerpt):
        {paper.full_text[:10000] if paper.full_text else 'Not available'}
        """
        return prompt
    
    def summarize(self, paper: Paper) -> str:
        """Summarize a paper using OpenAI's API."""
        prompt = self._create_prompt(paper)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            return summary
        except Exception as e:
            logger.error(f"Error using OpenAI API: {str(e)}")
            return f"Error generating summary: {str(e)}"
            
class DeepSeekSummarizer(LLMSummarizer):
    """Summarizer using DeepSeek's API."""
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _create_prompt(self, paper: Paper) -> str:
        """Create a prompt for the LLM."""
        prompt = f"""
        Please provide a comprehensive summary of the following research paper. Structure your summary with these sections:
        
        1. Key Findings: The main discoveries or contributions in 3-5 bullet points
        2. Research Question/Problem: What problem is the paper trying to solve?
        3. Methodology: How did they approach the problem?
        4. Results: What were the outcomes?
        5. Implications: Why these findings matter for the field
        
        Here's the paper information:
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Published: {paper.published_date or 'Not specified'}
        
        Abstract:
        {paper.abstract}
        
        Full Text (excerpt):
        {paper.full_text[:10000] if paper.full_text else 'Not available'}
        """
        return prompt
    
    def summarize(self, paper: Paper) -> str:
        """Summarize a paper using DeepSeek's API."""
        prompt = self._create_prompt(paper)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.1  # Lower temperature for more factual summaries
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            return summary
        except Exception as e:
            logger.error(f"Error using DeepSeek API: {str(e)}")
            return f"Error generating summary: {str(e)}"

class ResearchPaperSummarizer:
    """Main class to orchestrate the paper fetching and summarization process."""
    def __init__(self, llm_summarizer: LLMSummarizer, paper_fetcher: PaperFetcher = None):
        self.llm_summarizer = llm_summarizer
        self.paper_fetcher = paper_fetcher or ArxivFetcher()
    
    def search_and_summarize(self, query: str, max_results: int = 5, download_full_text: bool = True) -> List[Paper]:
        """Search for papers and summarize them."""
        # Get papers
        papers = self.paper_fetcher.fetch_papers(query, max_results)
        
        # Download full text if requested
        if download_full_text:
            logger.info("Downloading full text for papers")
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(tqdm(
                    executor.map(self.paper_fetcher.fetch_paper_content, papers),
                    total=len(papers),
                    desc="Downloading papers"
                ))
            papers = results
        
        # Summarize papers
        logger.info("Summarizing papers")
        for i, paper in enumerate(tqdm(papers, desc="Summarizing")):
            summary = self.llm_summarizer.summarize(paper)
            paper.summary = summary
            # Rate limiting to avoid API throttling
            if i < len(papers) - 1:
                time.sleep(1)
        
        return papers
    
    def save_summaries(self, papers: List[Paper], output_dir: str = "summaries"):
        """Save paper summaries to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for paper in papers:
            # Create a safe filename from the title
            safe_title = re.sub(r'[^\w\s-]', '', paper.title)
            safe_title = re.sub(r'[\s-]+', '_', safe_title).strip('_')
            filename = os.path.join(output_dir, f"{safe_title[:100]}.md")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {paper.title}\n\n")
                f.write(f"**Authors:** {', '.join(paper.authors)}\n\n")
                f.write(f"**Published:** {paper.published_date or 'Not specified'}\n\n")
                f.write(f"**URL:** {paper.url}\n\n")
                f.write("## Abstract\n\n")
                f.write(f"{paper.abstract}\n\n")
                f.write("## Summary\n\n")
                f.write(f"{paper.summary}\n\n")
        
        logger.info(f"Saved {len(papers)} summaries to {output_dir}")
    
    def summarize_from_pdf(self, pdf_path: str) -> Paper:
        """Summarize a paper from a local PDF file."""
        try:
            # Extract text from PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            # Clean up the text
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Try to extract title and authors from the first page
            first_page = pdf_reader.pages[0].extract_text()
            
            # Create a Paper object with extracted information
            paper = Paper(
                title=os.path.basename(pdf_path),  # Just use filename as fallback
                authors=["Unknown"],
                abstract="",
                url="",
                full_text=text
            )
            
            # Summarize
            summary = self.llm_summarizer.summarize(paper)
            paper.summary = summary
            
            return paper
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

# Example usage
def main():
    # Configure API keys (use appropriate key based on provider selection)
    api_key = os.environ.get("API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()
    
    if not api_key:
        raise ValueError("API_KEY environment variable is not set")
    
    # Create components based on selected provider
    if provider == "anthropic":
        summarizer = AnthropicSummarizer(api_key=api_key)
    elif provider == "openai":
        summarizer = OpenAISummarizer(api_key=api_key)
    elif provider == "deepseek":
        summarizer = DeepSeekSummarizer(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose anthropic, openai, or deepseek")
    
    fetcher = ArxivFetcher()
    
    # Create the main orchestrator
    paper_summarizer = ResearchPaperSummarizer(
        llm_summarizer=summarizer,
        paper_fetcher=fetcher
    )
    
    # Get search query from command line or use default
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Large language models"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"Searching for: {query}")
    print(f"Using {provider.capitalize()} for summarization")
    
    # Search for papers on a topic
    papers = paper_summarizer.search_and_summarize(
        query=query,
        max_results=max_results,
        download_full_text=True
    )
    
    # Save summaries
    paper_summarizer.save_summaries(papers)
    
    # Print summary of the first paper
    if papers:
        print(f"\nTitle: {papers[0].title}")
        print(f"Authors: {', '.join(papers[0].authors)}")
        print("\nSummary:")
        print(papers[0].summary)
        print(f"\nSaved {len(papers)} summaries to the 'summaries' directory")

if __name__ == "__main__":
    main()
