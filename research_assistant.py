import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Core libraries
import openai
import faiss
import numpy as np
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Additional utilities
import requests
from bs4 import BeautifulSoup
import arxiv
import streamlit as st
from pathlib import Path
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchQuery:
    """Structure for research queries"""
    query: str
    topic: str
    depth: str = "moderate"  # surface, moderate, deep
    sources: List[str] = None
    timeframe: str = "recent"  # all, recent, last_year
    
class VectorDatabase:
    """Quantized FAISS vector database for semantic search"""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.documents = []
        self.index_path = "research_index"
        
    def create_index(self, documents: List[Document]):
        """Create quantized FAISS index from documents"""
        try:
            if documents:
                # Create FAISS vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Get the index and quantize it
                index = self.vector_store.index
                
                # Create quantized index for better performance
                quantizer = faiss.IndexFlatL2(index.d)
                quantized_index = faiss.IndexIVFPQ(quantizer, index.d, 100, 8, 8)
                quantized_index.train(index.reconstruct_n(0, index.ntotal))
                quantized_index.add(index.reconstruct_n(0, index.ntotal))
                
                # Replace with quantized index
                self.vector_store.index = quantized_index
                
                logger.info(f"Created quantized FAISS index with {len(documents)} documents")
                return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def save_index(self):
        """Save the vector database"""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
            logger.info("Vector database saved")
    
    def load_index(self):
        """Load existing vector database"""
        try:
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
            logger.info("Vector database loaded")
            return True
        except:
            logger.info("No existing index found")
            return False
    
    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """High-recall semantic search"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []

class DocumentProcessor:
    """Process and chunk documents for the vector database"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF documents"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Process web content"""
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(self.text_splitter.split_documents(docs))
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
        return documents
    
    def process_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """Process raw text"""
        doc = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([doc])

class ArxivAgent:
    """Agent for searching ArXiv papers"""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search ArXiv for research papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in self.client.results(search):
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'url': result.pdf_url,
                    'published': result.published,
                    'categories': result.categories
                })
            
            logger.info(f"Found {len(papers)} papers on ArXiv")
            return papers
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []

class ScholarAgent:
    """Agent for searching Google Scholar"""
    
    def __init__(self):
        pass
    
    def search_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search Google Scholar for papers"""
        try:
            papers = []
            search_query = sch.search_pubs(query)
            
            count = 0
            for pub in search_query:
                if count >= max_results:
                    break
                    
                papers.append({
                    'title': pub.get('title', ''),
                    'authors': pub.get('author', ''),
                    'summary': pub.get('abstract', ''),
                    'url': pub.get('pub_url', ''),
                    'citations': pub.get('num_citations', 0),
                    'year': pub.get('year', '')
                })
                count += 1
            
            logger.info(f"Found {len(papers)} papers on Scholar")
            return papers
        except Exception as e:
            logger.error(f"Error searching Scholar: {e}")
            return []

class WebSearchAgent:
    """Agent for web search and content extraction"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_web(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search web for relevant content"""
        try:
            # This would typically use a search API like Google Custom Search
            # For demonstration, we'll use a simple approach
            search_results = []
            
            # You can integrate with search APIs here
            # For now, returning placeholder structure
            logger.info(f"Web search completed for: {query}")
            return search_results
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    def extract_content(self, url: str) -> str:
        """Extract content from web page"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit content length
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""

class SynthesisAgent:
    """Agent for synthesizing research findings"""
    
    def __init__(self, llm):
        self.llm = llm
        self.synthesis_prompt = PromptTemplate(
            input_variables=["research_data", "query"],
            template="""
            Based on the following research data, provide a comprehensive synthesis for the query: {query}

            Research Data:
            {research_data}

            Please provide:
            1. Key findings and patterns
            2. Contradictions or debates in the literature
            3. Research gaps identified
            4. Implications and future directions
            5. Summary of most relevant sources

            Synthesis:
            """
        )
    
    def synthesize(self, research_data: str, query: str) -> str:
        """Synthesize research findings"""
        try:
            prompt = self.synthesis_prompt.format(
                research_data=research_data,
                query=query
            )
            response = self.llm(prompt)
            return response
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return "Error generating synthesis"

class MultiAgentResearchAssistant:
    """Main orchestrator for multi-agent research"""
    
    def __init__(self, openai_api_key: str):
        # Initialize OpenAI
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LLM
        self.llm = OpenAI(temperature=0.7, max_tokens=2000)
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.doc_processor = DocumentProcessor()
        self.arxiv_agent = ArxivAgent()
        self.scholar_agent = ScholarAgent()
        self.web_agent = WebSearchAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Load existing index if available
        self.vector_db.load_index()
        
        logger.info("Multi-Agent Research Assistant initialized")
    
    async def research(self, query: ResearchQuery) -> Dict[str, Any]:
        """Conduct comprehensive research"""
        results = {
            'query': query.query,
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'arxiv': [],
                'scholar': [],
                'web': [],
                'documents': []
            },
            'synthesis': '',
            'recommendations': []
        }
        
        try:
            # Phase 1: Gather information from multiple sources
            logger.info("Phase 1: Gathering information...")
            
            # ArXiv search
            if query.sources is None or 'arxiv' in query.sources:
                results['sources']['arxiv'] = self.arxiv_agent.search_papers(
                    query.query, max_results=10
                )
            
            # Scholar search
            if query.sources is None or 'scholar' in query.sources:
                results['sources']['scholar'] = self.scholar_agent.search_papers(
                    query.query, max_results=10
                )
            
            # Web search
            if query.sources is None or 'web' in query.sources:
                results['sources']['web'] = self.web_agent.search_web(
                    query.query, max_results=10
                )
            
            # Phase 2: Process and index new content
            logger.info("Phase 2: Processing content...")
            
            new_documents = []
            
            # Process ArXiv papers
            for paper in results['sources']['arxiv']:
                doc_text = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nSummary: {paper['summary']}"
                docs = self.doc_processor.process_text(doc_text, {
                    'source': 'arxiv',
                    'title': paper['title'],
                    'url': paper['url']
                })
                new_documents.extend(docs)
            
            # Process Scholar papers
            for paper in results['sources']['scholar']:
                doc_text = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nSummary: {paper['summary']}"
                docs = self.doc_processor.process_text(doc_text, {
                    'source': 'scholar',
                    'title': paper['title'],
                    'citations': paper['citations']
                })
                new_documents.extend(docs)
            
            # Update vector database
            if new_documents:
                self.vector_db.create_index(new_documents)
                self.vector_db.save_index()
            
            # Phase 3: Semantic search and retrieval
            logger.info("Phase 3: Semantic search...")
            
            relevant_docs = self.vector_db.similarity_search(query.query, k=20)
            
            # Phase 4: Synthesis
            logger.info("Phase 4: Synthesizing findings...")
            
            research_data = ""
            for doc in relevant_docs:
                research_data += f"\n\n--- Document ---\n{doc.page_content}"
                if doc.metadata:
                    research_data += f"\nMetadata: {doc.metadata}"
            
            results['synthesis'] = self.synthesis_agent.synthesize(
                research_data, query.query
            )
            
            # Phase 5: Generate recommendations
            results['recommendations'] = self._generate_recommendations(
                results['sources'], query.query
            )
            
            logger.info("Research completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in research process: {e}")
            results['error'] = str(e)
            return results
    
    def _generate_recommendations(self, sources: Dict, query: str) -> List[str]:
        """Generate research recommendations"""
        recommendations = []
        
        # Analyze sources for recommendations
        total_papers = len(sources['arxiv']) + len(sources['scholar'])
        
        if total_papers > 15:
            recommendations.append("Rich literature available - consider narrowing scope for deeper analysis")
        elif total_papers < 5:
            recommendations.append("Limited sources found - consider broadening search terms")
        
        # Check for recent papers
        recent_arxiv = [p for p in sources['arxiv'] if 
                       (datetime.now() - p['published']).days < 365]
        
        if len(recent_arxiv) < 3:
            recommendations.append("Consider searching for more recent publications")
        
        recommendations.append("Review synthesis for research gaps and future directions")
        
        return recommendations
    
    def add_documents(self, file_paths: List[str]):
        """Add documents to the knowledge base"""
        all_docs = []
        
        for path in file_paths:
            if path.endswith('.pdf'):
                docs = self.doc_processor.process_pdf(path)
                all_docs.extend(docs)
        
        if all_docs:
            self.vector_db.create_index(all_docs)
            self.vector_db.save_index()
            logger.info(f"Added {len(all_docs)} document chunks to knowledge base")

# Streamlit Interface
def create_streamlit_interface():
    """Create Streamlit interface for the research assistant"""
    
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 AI-Powered Multi-Agent Research Assistant")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Try to get API key from Streamlit secrets first (for deployed app), then from environment
    api_key = None
    
    # Check Streamlit secrets (for cloud deployment)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except:
        pass
    
    # If not in secrets, try environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Only show input field if no API key found in secrets or environment
    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
        st.info("💡 For security, set OPENAI_API_KEY in Streamlit secrets when deploying.")
        return
    else:
        st.sidebar.success("✅ API Key loaded from secure storage")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MultiAgentResearchAssistant(api_key)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Research Query")
        
        # Query input
        query_text = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are the latest developments in transformer architecture optimization for NLP tasks?"
        )
        
        # Options
        col1a, col1b = st.columns(2)
        with col1a:
            depth = st.selectbox("Research Depth", ["surface", "moderate", "deep"])
            timeframe = st.selectbox("Timeframe", ["recent", "last_year", "all"])
        
        with col1b:
            sources = st.multiselect(
                "Sources",
                ["arxiv", "scholar", "web"],
                default=["arxiv", "scholar"]
            )
        
        # File upload
        st.subheader("Upload Documents (Optional)")
        uploaded_files = st.file_uploader(
            "Upload PDF files to add to knowledge base",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("🔍 Start Research", type="primary"):
            if query_text:
                with st.spinner("Conducting research..."):
                    # Create research query
                    research_query = ResearchQuery(
                        query=query_text,
                        topic=query_text,
                        depth=depth,
                        sources=sources,
                        timeframe=timeframe
                    )
                    
                    # Run research
                    results = asyncio.run(
                        st.session_state.assistant.research(research_query)
                    )
                    
                    st.session_state.results = results
            else:
                st.error("Please enter a research query.")
    
    with col2:
        st.header("Quick Stats")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display stats
            arxiv_count = len(results['sources']['arxiv'])
            scholar_count = len(results['sources']['scholar'])
            web_count = len(results['sources']['web'])
            
            st.metric("ArXiv Papers", arxiv_count)
            st.metric("Scholar Papers", scholar_count)
            st.metric("Web Sources", web_count)
            
            # Show recommendations
            st.subheader("Recommendations")
            for rec in results['recommendations']:
                st.info(rec)
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.markdown("---")
        st.header("Research Results")
        
        # Synthesis
        st.subheader("📋 Research Synthesis")
        st.write(results['synthesis'])
        
        # Detailed sources
        tabs = st.tabs(["ArXiv Papers", "Scholar Papers", "Web Sources"])
        
        with tabs[0]:
            for paper in results['sources']['arxiv']:
                with st.expander(paper['title']):
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Categories:** {', '.join(paper['categories'])}")
                    st.write(f"**Summary:** {paper['summary']}")
                    st.write(f"**URL:** {paper['url']}")
        
        with tabs[1]:
            for paper in results['sources']['scholar']:
                with st.expander(paper['title']):
                    st.write(f"**Authors:** {paper['authors']}")
                    st.write(f"**Year:** {paper['year']}")
                    st.write(f"**Citations:** {paper['citations']}")
                    st.write(f"**Summary:** {paper['summary']}")
                    if paper['url']:
                        st.write(f"**URL:** {paper['url']}")
        
        with tabs[2]:
            for source in results['sources']['web']:
                with st.expander(source.get('title', 'Web Source')):
                    st.write(source.get('summary', 'No summary available'))
                    if source.get('url'):
                        st.write(f"**URL:** {source['url']}")

# Main execution
if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        create_streamlit_interface()
    except ImportError:
        # Run as script
        print("AI-Powered Multi-Agent Research Assistant")
        print("Install streamlit to run the web interface: pip install streamlit")
        
        # Example usage
        api_key = input("Enter your OpenAI API key: ")
        assistant = MultiAgentResearchAssistant(api_key)
        
        query = ResearchQuery(
            query="transformer architecture optimization techniques",
            topic="machine learning",
            depth="moderate"
        )
        
        results = asyncio.run(assistant.research(query))
        print("\nResearch Results:")
        print("="*50)
        print(results['synthesis'])
