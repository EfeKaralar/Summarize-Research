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
    def fetch_papers(self, query: str, max_results: int = 10, sort_by_date: bool = False) -> List[Paper]:
        """
        Fetch papers from arXiv based on a query.
        
        Args:
            query (str): Search query for papers
            max_results (int): Maximum number of results to return
            sort_by_date (bool): If True, sort results by submission date (newest first)
        
        Returns:
            List[Paper]: List of found papers
        """
        logger.info(f"Fetching papers from arXiv with query: {query}")
        
        # Determine sort criteria
        sort_criteria = arxiv.SortCriterion.SubmittedDate if sort_by_date else arxiv.SortCriterion.Relevance
        
        # Handle both author and field queries (author queries always sort by date)
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
                sort_by=sort_criteria
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
        
        logger.info(f"Found {len(papers)} papers on arXiv{' (sorted by date)' if sort_by_date else ''}")
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
        self.session_id = time.strftime("%Y%m%d_%H%M%S")  # Generate a unique session ID based on timestamp
    
    def search_and_summarize(self, query: str, max_results: int = 5, download_full_text: bool = True, sort_by_date: bool = False) -> List[Paper]:
        """
        Search for papers and summarize them.
        
        Args:
            query (str): Search query for papers
            max_results (int): Maximum number of results to return
            download_full_text (bool): Whether to download and extract the full text from PDFs
            sort_by_date (bool): If True, sort results by submission date (newest first)
        
        Returns:
            List[Paper]: List of papers with summaries
        """
        # Get papers
        papers = self.paper_fetcher.fetch_papers(query, max_results, sort_by_date)
        
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
    
    def get_session_folder(self, base_dir: str = "summaries") -> str:
        """
        Get the folder for the current session.
        
        Args:
            base_dir (str): Base directory for all summary sessions
            
        Returns:
            str: Path to the session folder
        """
        # Create a safe folder name that includes the session ID and a timestamp
        session_folder = os.path.join(base_dir, self.session_id)
        os.makedirs(session_folder, exist_ok=True)
        return session_folder
    
    def save_summaries(self, papers: List[Paper], output_dir: str = None):
        """
        Save paper summaries to files in a session-specific folder.
        
        Args:
            papers (List[Paper]): List of papers with summaries
            output_dir (str, optional): Custom output directory. If None, uses session folder.
        """
        # If no custom output directory provided, use session folder
        if output_dir is None:
            output_dir = self.get_session_folder()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create session info file with metadata
        with open(os.path.join(output_dir, "_session_info.md"), 'w', encoding='utf-8') as f:
            f.write(f"# Research Summary Session\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Number of Papers:** {len(papers)}\n\n")
            f.write("## Papers in this Session\n\n")
            for paper in papers:
                f.write(f"- {paper.title} ({', '.join(paper.authors[:3])}{', et al.' if len(paper.authors) > 3 else ''})\n")
        
        # Save individual paper summaries
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
    import argparse
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Research Paper Summarizer")
    parser.add_argument("query", nargs="?", default="Large language models", help="Search query for papers")
    parser.add_argument("-n", "--num-results", type=int, default=3, help="Number of papers to fetch (default: 3)")
    parser.add_argument("-s", "--sort-by-date", action="store_true", help="Sort results by date (newest first)")
    parser.add_argument("-o", "--output-dir", help="Custom output directory")
    parser.add_argument("-p", "--provider", default=os.environ.get("LLM_PROVIDER", "deepseek"), 
                        choices=["anthropic", "openai", "deepseek"], help="LLM provider to use")
    parser.add_argument("-f", "--full-text", action="store_true", default=True, 
                        help="Download and use full text for summarization")
    parser.add_argument("-a", "--api-key", help="API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Configure API keys (use appropriate key based on provider selection)
    api_key = args.api_key or os.environ.get("API_KEY")
    provider = args.provider.lower()
    
    if not api_key:
        raise ValueError("API key must be provided either via --api-key or API_KEY environment variable")
    
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
    
    print(f"Research Paper Summarizer")
    print(f"------------------------")
    print(f"Searching for: {args.query}")
    print(f"Number of papers: {args.num_results}")
    print(f"Sort by date: {'Yes' if args.sort_by_date else 'No'}")
    print(f"Using {provider.capitalize()} for summarization")
    
    # Search for papers on a topic
    papers = paper_summarizer.search_and_summarize(
        query=args.query,
        max_results=args.num_results,
        download_full_text=args.full_text,
        sort_by_date=args.sort_by_date
    )
    
    # Save summaries
    output_dir = paper_summarizer.save_summaries(papers, args.output_dir)
    
    # Print summary of the first paper
    if papers:
        print(f"\nTitle: {papers[0].title}")
        print(f"Authors: {', '.join(papers[0].authors)}")
        print("\nSummary excerpt:")
        # Print just the first few lines of the summary
        summary_lines = papers[0].summary.split('\n')
        print('\n'.join(summary_lines[:5]) + ('...' if len(summary_lines) > 5 else ''))
        
        # Get the session folder for output message
        session_folder = args.output_dir or paper_summarizer.get_session_folder()
        print(f"\nSaved {len(papers)} summaries to: {session_folder}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()