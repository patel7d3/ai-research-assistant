import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Core libs
import openai
import faiss
import numpy as np

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document  # safer for langchain==1.x

import requests
from bs4 import BeautifulSoup
import arxiv
import streamlit as st
from pathlib import Path
import pickle
from dotenv import load_dotenv

# Try to support Google Scholar, but don’t die if package isn’t installed
try:
    from scholarly import scholarly as sch  # pip install scholarly
except Exception:
    sch = None

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """Structure for research queries"""
    query: str
    topic: str
    depth: str = "moderate"        # "surface", "moderate", "deep"
    sources: List[str] = None      # ["arxiv", "scholar", "web"]
    timeframe: str = "recent"      # "all", "recent", "last_year"


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
                # build base FAISS from docs
                self.vector_store = FAISS.from_documents(documents, self.embeddings)

                # grab raw index
                index = self.vector_store.index

                # optional quantization step
                quantizer = faiss.IndexFlatL2(index.d)
                quantized_index = faiss.IndexIVFPQ(
                    quantizer,
                    index.d,
                    100,   # nlist
                    8,     # m
                    8      # nbits
                )

                # train / add
                quantized_index.train(index.reconstruct_n(0, index.ntotal))
                quantized_index.add(index.reconstruct_n(0, index.ntotal))

                # swap in quantized index
                self.vector_store.index = quantized_index

                logger.info(f"Created quantized FAISS index with {len(documents)} documents")
                return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def save_index(self):
        """Persist the index locally on disk"""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
            logger.info("Vector database saved")

    def load_index(self):
        """Load an existing index if present"""
        try:
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
            logger.info("Vector database loaded")
            return True
        except Exception:
            logger.info("No existing index found")
            return False

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Semantic KNN search"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []


class DocumentProcessor:
    """Split PDFs / web pages / text into chunks for retrieval"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Read and chunk a PDF into Documents"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []

    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Fetch pages and chunk them"""
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
        """Turn raw text into split Documents"""
        doc = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([doc])


class ArxivAgent:
    """Lightweight ArXiv fetcher using python-arxiv"""

    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in self.client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "url": result.pdf_url,
                    "published": result.published,
                    "categories": result.categories,
                })
            logger.info(f"Found {len(papers)} papers on ArXiv")
            return papers
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []


class ScholarAgent:
    """Google Scholar fetcher (best-effort).
       If scholarly isn't installed, we just return [] instead of crashing.
    """

    def __init__(self):
        self.enabled = sch is not None

    def search_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        if not self.enabled:
            logger.warning("ScholarAgent unavailable (scholarly not installed)")
            return []

        try:
            papers = []
            search_query = sch.search_pubs(query)

            count = 0
            for pub in search_query:
                if count >= max_results:
                    break

                papers.append({
                    "title": pub.get("title", ""),
                    "authors": pub.get("author", ""),
                    "summary": pub.get("abstract", ""),
                    "url": pub.get("pub_url", ""),
                    "citations": pub.get("num_citations", 0),
                    "year": pub.get("year", ""),
                })
                count += 1

            logger.info(f"Found {len(papers)} papers on Scholar")
            return papers
        except Exception as e:
            logger.error(f"Error searching Scholar: {e}")
            return []


class WebSearchAgent:
    """Placeholder web search + HTML scrape"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def search_web(self, query: str, max_results: int = 10) -> List[Dict]:
        # TODO: plug in a real search API.
        logger.info(f"Web search completed for: {query}")
        return []

    def extract_content(self, url: str) -> str:
        """Naive HTML text extraction"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # strip scripts/styles
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text[:5000]
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""


class SynthesisAgent:
    """LLM that turns retrieved chunks into a narrative answer"""

    def __init__(self, llm):
        self.llm = llm
        self.synthesis_prompt = PromptTemplate(
            input_variables=["research_data", "query"],
            template=(
                "Based on the following research data, provide a comprehensive synthesis "
                "for the query: {query}\n\n"
                "Research Data:\n"
                "{research_data}\n\n"
                "Please provide:\n"
                "1. Key findings and patterns\n"
                "2. Contradictions or debates in the literature\n"
                "3. Research gaps identified\n"
                "4. Implications and future directions\n"
                "5. Summary of most relevant sources\n\n"
                "Synthesis:\n"
            ),
        )

    def synthesize(self, research_data: str, query: str) -> str:
        try:
            prompt = self.synthesis_prompt.format(
                research_data=research_data,
                query=query,
            )
            response = self.llm(prompt)
            return response
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return "Error generating synthesis"


class MultiAgentResearchAssistant:
    """Main orchestrator"""

    def __init__(self, openai_api_key: str):
        # wire OpenAI key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # LLM
        self.llm = OpenAI(temperature=0.7, max_tokens=2000)

        # Components
        self.vector_db = VectorDatabase()
        self.doc_processor = DocumentProcessor()
        self.arxiv_agent = ArxivAgent()
        self.scholar_agent = ScholarAgent()
        self.web_agent = WebSearchAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)

        # Chat memory (not deeply used yet, but fine to keep)
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # Try to load any saved FAISS index
        self.vector_db.load_index()

        logger.info("Multi-Agent Research Assistant initialized")

    async def research(self, query: ResearchQuery) -> Dict[str, Any]:
        """Full pipeline:
        1. gather sources
        2. embed/index
        3. retrieve
        4. synthesize
        5. recommendations
        """
        results = {
            "query": query.query,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "arxiv": [],
                "scholar": [],
                "web": [],
                "documents": [],
            },
            "synthesis": "",
            "recommendations": [],
        }

        try:
            # Phase 1: gather
            logger.info("Phase 1: Gathering information...")

            if query.sources is None or "arxiv" in query.sources:
                results["sources"]["arxiv"] = self.arxiv_agent.search_papers(
                    query.query, max_results=10
                )

            if query.sources is None or "scholar" in query.sources:
                results["sources"]["scholar"] = self.scholar_agent.search_papers(
                    query.query, max_results=10
                )

            if query.sources is None or "web" in query.sources:
                results["sources"]["web"] = self.web_agent.search_web(
                    query.query, max_results=10
                )

            # Phase 2: build/update the vector DB
            logger.info("Phase 2: Processing content...")

            new_documents = []

            # turn arxiv results into Documents
            for paper in results["sources"]["arxiv"]:
                doc_text = (
                    f"Title: {paper['title']}\n"
                    f"Authors: {', '.join(paper['authors'])}\n"
                    f"Summary: {paper['summary']}"
                )
                docs = self.doc_processor.process_text(
                    doc_text,
                    {
                        "source": "arxiv",
                        "title": paper["title"],
                        "url": paper["url"],
                    },
                )
                new_documents.extend(docs)

            # turn scholar results into Documents
            for paper in results["sources"]["scholar"]:
                doc_text = (
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"Summary: {paper['summary']}"
                )
                docs = self.doc_processor.process_text(
                    doc_text,
                    {
                        "source": "scholar",
                        "title": paper["title"],
                        "citations": paper.get("citations", 0),
                    },
                )
                new_documents.extend(docs)

            if new_documents:
                self.vector_db.create_index(new_documents)
                self.vector_db.save_index()

            # Phase 3: retrieve
            logger.info("Phase 3: Semantic search...")

            relevant_docs = self.vector_db.similarity_search(query.query, k=20)

            # Phase 4: synthesize
            logger.info("Phase 4: Synthesizing findings...")

            research_data = ""
            for doc in relevant_docs:
                research_data += f"\n\n--- Document ---\n{doc.page_content}"
                if doc.metadata:
                    research_data += f"\nMetadata: {doc.metadata}"

            results["synthesis"] = self.synthesis_agent.synthesize(
                research_data, query.query
            )

            # Phase 5: recommendations
            results["recommendations"] = self._generate_recommendations(
                results["sources"], query.query
            )

            logger.info("Research completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in research process: {e}")
            results["error"] = str(e)
            return results

    def _generate_recommendations(self, sources: Dict, query: str) -> List[str]:
        """Simple heuristics to guide next steps"""
        recommendations = []

        total_papers = len(sources["arxiv"]) + len(sources["scholar"])
        if total_papers > 15:
            recommendations.append(
                "Rich literature available — consider narrowing scope for deeper analysis"
            )
        elif total_papers < 5:
            recommendations.append(
                "Limited sources found — consider broadening search terms"
            )

        # check for 'recent' work (< 1 year old)
        recent_arxiv = [
            p for p in sources["arxiv"]
            if (datetime.now() - p["published"]).days < 365
        ]
        if len(recent_arxiv) < 3:
            recommendations.append(
                "Consider searching for more recent publications"
            )

        recommendations.append(
            "Review synthesis for research gaps and future directions"
        )

        return recommendations


def create_streamlit_interface():
    """Streamlit UI wrapper"""

    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 AI-Powered Multi-Agent Research Assistant")
    st.markdown("---")

    # Sidebar config
    st.sidebar.header("Configuration")

    # Load key from secrets or env first so we don't force users to paste every time
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys",
        )
        st.warning("Please enter your OpenAI API Key in the sidebar.")
        st.info("For deployment, set OPENAI_API_KEY in Streamlit secrets.")
        return
    else:
        st.sidebar.success("API Key loaded from secure storage")

    # Initialize assistant once per session
    if "assistant" not in st.session_state:
        st.session_state.assistant = MultiAgentResearchAssistant(api_key)

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Research Query")

        query_text = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are the latest developments in transformer architecture optimization for NLP tasks?",
        )

        col1a, col1b = st.columns(2)
        with col1a:
            depth = st.selectbox("Research Depth", ["surface", "moderate", "deep"])
            timeframe = st.selectbox("Timeframe", ["recent", "last_year", "all"])
        with col1b:
            sources = st.multiselect(
                "Sources",
                ["arxiv", "scholar", "web"],
                default=["arxiv", "scholar"],
            )

        st.subheader("Upload Documents (Optional)")
        uploaded_files = st.file_uploader(
            "Upload PDF files to add to knowledge base",
            type=["pdf"],
            accept_multiple_files=True,
        )
        # TODO: actually add uploaded PDFs into vector DB

        if st.button("🔍 Start Research", type="primary"):
            if query_text:
                with st.spinner("Conducting research..."):
                    research_query = ResearchQuery(
                        query=query_text,
                        topic=query_text,
                        depth=depth,
                        sources=sources,
                        timeframe=timeframe,
                    )

                    results = asyncio.run(
                        st.session_state.assistant.research(research_query)
                    )

                    st.session_state.results = results
            else:
                st.error("Please enter a research query.")

    with col2:
        st.header("Quick Stats")

        if "results" in st.session_state:
            results = st.session_state.results

            arxiv_count = len(results["sources"]["arxiv"])
            scholar_count = len(results["sources"]["scholar"])
            web_count = len(results["sources"]["web"])

            st.metric("ArXiv Papers", arxiv_count)
            st.metric("Scholar Papers", scholar_count)
            st.metric("Web Sources", web_count)

            st.subheader("Recommendations")
            for rec in results["recommendations"]:
                st.info(rec)

    # Results section
    if "results" in st.session_state:
        results = st.session_state.results

        st.markdown("---")
        st.header("Research Results")

        st.subheader("📋 Research Synthesis")
        st.write(results["synthesis"])

        tabs = st.tabs(["ArXiv Papers", "Scholar Papers", "Web Sources"])

        with tabs[0]:
            for paper in results["sources"]["arxiv"]:
                with st.expander(paper["title"]):
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Categories:** {', '.join(paper['categories'])}")
                    st.write(f"**Summary:** {paper['summary']}")
                    st.write(f"**URL:** {paper['url']}")

        with tabs[1]:
            for paper in results["sources"]["scholar"]:
                with st.expander(paper["title"]):
                    st.write(f"**Authors:** {paper['authors']}")
                    st.write(f"**Year:** {paper['year']}")
                    st.write(f"Citations:** {paper['citations']}")
                    st.write(f"**Summary:** {paper['summary']}")
                    if paper["url"]:
                        st.write(f"**URL:** {paper['url']}")

        with tabs[2]:
            for source in results["sources"]["web"]:
                with st.expander(source.get("title", "Web Source")):
                    st.write(source.get("summary", "No summary available"))
                    if source.get("url"):
                        st.write(f"**URL:** {source['url']}")


if __name__ == "__main__":
    # try to run Streamlit UI
    try:
        import streamlit as st  # noqa: F401
        create_streamlit_interface()
    except ImportError:
        # CLI fallback
        print("AI-Powered Multi-Agent Research Assistant")
        print("Install streamlit to run the web interface: pip install streamlit")

        api_key = input("Enter your OpenAI API key: ")
        assistant = MultiAgentResearchAssistant(api_key)

        query = ResearchQuery(
            query="transformer architecture optimization techniques",
            topic="machine learning",
            depth="moderate",
        )

        results = asyncio.run(assistant.research(query))
        print("\nResearch Results:")
        print("=" * 50)
        print(results["synthesis"])
