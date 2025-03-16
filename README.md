# Research Paper Summarizer

A powerful tool for researchers, academics, and knowledge workers to efficiently process and understand scientific literature. This tool fetches research papers from arXiv based on your search criteria, generates comprehensive summaries using AI, and provides insightful comparative analysis of related papers.

## Features

- **Automated Paper Discovery**: Search arXiv by topic or author
- **AI-Powered Summarization**: Generate structured summaries with key findings, methodology, and implications
- **Comparative Analysis**: Understand connections between papers in a research area
- **Multiple LLM Support**: Use Claude, GPT, or DeepSeek models
- **Date-Based Sorting**: Prioritize recent research or most relevant papers
- **Session Management**: Organizes summaries in topic-specific folders
- **Full-Text Processing**: Extracts and analyzes entire paper contents
- **Markdown Output**: Clean, structured summaries ready for reference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EfeKaralar/Summarize-Research.git
   cd Summarize-Research
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key as an environment variable:
   ```bash
   export API_KEY="your_api_key_here"
   export LLM_PROVIDER="deepseek"  # or "anthropic" or "openai"
   ```

## Usage

### Basic Search

Search for papers on a topic and generate summaries:

```bash
python research-summarizer.py search "quantum computing"
```

### Advanced Options

```bash
python research-summarizer.py search "reinforcement learning" -n 5 -s -p openai
```

Options:
- `-n, --num-results`: Number of papers to fetch (default: 3)
- `-s, --sort-by-date`: Sort by submission date (newest first)
- `-o, --output-dir`: Custom output directory
- `-p, --provider`: LLM provider to use (anthropic/openai/deepseek)
- `-a, --api-key`: API key (overrides environment variable)
- `-f, --full-text`: Download and use full text for summarization (default: true)

### Search by Author

Find and summarize papers by a specific researcher:

```bash
python research-summarizer.py search "author:Geoffrey Hinton" -n 5 -s
```

### Generate Comparative Analysis

For an existing directory of paper summaries:

```bash
python research-summarizer.py analyze summaries/20250315_quantum_computing
```

## Output Structure

Each search creates a timestamped directory with the query in the name (e.g., `summaries/20250315_145030_quantum_computing/`):

- `_session_info.md`: Overview of the search session
- `_comparative_analysis.md`: Analysis of how papers relate to each other
- Individual paper summaries as markdown files

### Sample Summary Structure

Each paper summary includes:
- Title, authors, and publication date
- Abstract
- Key findings (bullet points)
- Research question/problem
- Methodology
- Results
- Implications for the field

## How It Works

1. **Search**: Uses the arXiv API to find papers matching your query
2. **Download**: Retrieves and extracts text from PDFs
3. **Summarize**: Processes each paper with an LLM to create structured summaries
4. **Analyze**: Generates a comparative analysis of the paper collection
5. **Output**: Saves everything as well-formatted markdown files

## LLM Integration

The tool supports three LLM providers:

- **Anthropic Claude**: Well-suited for nuanced academic content
- **OpenAI GPT**: Excellent general-purpose summarization
- **DeepSeek**: Cost-effective alternative with solid performance

## Requirements

- Python 3.8+
- Required packages: arxiv, PyPDF2, requests, tqdm
- API key for at least one provider (Anthropic, OpenAI, or DeepSeek)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Improvements

- Add support for more paper sources (PubMed, IEEE, ACM, etc.)
- Implement citation graph analysis
- Add visualization of research trends
- Enable batch processing of large paper collections
