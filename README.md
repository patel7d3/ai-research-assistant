# AI-Powered Multi-Agent Research Assistant

An intelligent research assistant that leverages Large Language Models (LLMs), vector databases, and retrieval-augmented generation (RAG) to automate academic research processes.

## Features

- **Multi-Agent Architecture**: Specialized agents for different research tasks
- **Quantized FAISS Vector Database**: High-performance semantic search
- **Retrieval-Augmented Generation (RAG)**: Combines semantic search with GPT-based synthesis
- **ArXiv Integration**: Automated paper discovery from ArXiv
- **Intelligent Synthesis**: LLM-powered analysis and summary generation

## Tech Stack

- **AI/ML**: OpenAI GPT, LangChain, FAISS
- **Data Processing**: Python, Pandas, NumPy
- **Interface**: Streamlit
- **APIs**: ArXiv API

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/patel7d3/ai-research-assistant.git
cd ai-research-assistant
```

2. Create a virtual environment:
```bash
python -m venv research_env
source research_env/bin/activate  # On Windows: research_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running Locally

```bash
streamlit run research_assistant.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. Enter your research question in the text area
2. Select research depth (surface, moderate, or deep)
3. Choose timeframe for papers
4. Click "Start Research"
5. View synthesized results and discovered papers

## Live Demo

Try the live version: [AI Research Assistant](https://ai-research-assistant-patel7d3.streamlit.app)

## Project Structure

```
ai-research-assistant/
├── research_assistant.py    # Main application
├── requirements.txt          # Dependencies
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## How It Works

1. **Information Gathering**: Searches ArXiv for relevant papers
2. **Document Processing**: Chunks and creates embeddings
3. **Semantic Search**: Uses FAISS for high-recall retrieval
4. **Synthesis**: GPT generates comprehensive analysis
5. **Recommendations**: Provides research suggestions

## Impact

- **Time Savings**: Eliminates 40+ hours of manual research monthly
- **Comprehensive Coverage**: Automated multi-source search
- **Intelligent Analysis**: Context-aware synthesis
- **Scalable**: Handles complex research queries efficiently

## Limitations

- Currently supports ArXiv only (Google Scholar disabled due to reliability issues)
- Requires OpenAI API key (incurs costs per query)
- Vector database rebuilds for each session

## Future Enhancements

- [ ] Add support for additional academic databases
- [ ] Implement persistent vector storage
- [ ] Add PDF upload functionality
- [ ] Export research reports to PDF
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Devansh Patel
- Email: patel7d3@mail.uc.edu
- LinkedIn: [linkedin.com/in/dewanshpatel](https://linkedin.com/in/dewanshpatel)
- GitHub: [github.com/patel7d3](https://github.com/patel7d3)

## Acknowledgments

- Built as part of MS Business Analytics capstone project
- University of Cincinnati, Carl H. Lindner College of Business
- Technologies: OpenAI, LangChain, FAISS, Streamlit

## Support

For issues or questions, please open an issue on GitHub or contact me directly.
