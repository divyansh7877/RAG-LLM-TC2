#!/usr/bin/env python3
"""
Query expert‑call transcript embeddings stored in LanceDB with LlamaIndex and
served them through a Gradio UI with Meta's LLM.

Make sure your **document embeddings were produced with the *same* embedding
model** (INT8‑quantised `Alibaba-NLP/gte-large-en-v1.5`).
"""
import os
import multiprocessing as mp
import lancedb
import gradio as gr
import threading
from typing import Dict, Any

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import VectorStoreInfo, MetadataFilters, ExactMatchFilter
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./multi_user_db.lance"
TABLE_NAME = "document_embeddings"
EMBED_MODEL_NAME = "./models/gte-large-en-v1.5"  # INT8‐quantised, CPU
#GGUF_MODEL_PATH = "./models/models--lmutimb--Meta-Llama-3.1-8B-Instruct-Q5_0-GGUF/snapshots/0bec7ad9d7a9cfbd8a6d2100f10e114bec3e44ad/meta-llama-3.1-8b-instruct-q5_0.gguf"
GGUF_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"  # llama.cpp model\

has_cuda = torch.cuda.is_available()

# Runtime knobs
device = "cuda" if has_cuda else "cpu"  
BEAM_K = 30      # initial ANN beam for MMR
FINAL_K = 8     # chunks passed to the LLM
MMR_LAMBDA = 0.1
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
N_THREADS = mp.cpu_count()
N_GPU_LAYERS = -1 if has_cuda else 0 # set 0 if no GPU / VRAM < 12 GB
N_BATCH = 1024 if has_cuda else 64   # llama.cpp prompt batch size
PROMPT_CACHE_DIR = "./prompt_cache"

# ---------------------------------------------------------------------------
# Prompt enforcing source citations
# ---------------------------------------------------------------------------
QA_TEMPLATE = (
    "You are an AI assistant specialised in answering questions from provided expert‑call transcripts.\nUse the context but do not just provide the context, use the CONTEXT and the QUESTION to generate a meaningful answer. If the context lacks the answer, reply: \"The provided context does not contain information to answer this question.\"\n. When citing, follow this format: (Source: {document_name}, Page: {page_number}).\n"
    "------------------------\n"
    "CONTEXT:\n{context_str}\n"
    "------------------------\n"
    "QUESTION: {query_str}\n"
    "------------------------\n"
    "ANSWER:\n"
)

# ---------------------------------------------------------------------------
# Thread-safe Query Engine Factory
# ---------------------------------------------------------------------------

class QueryEngineFactory:
    """Thread-safe factory for creating query engines with proper isolation."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._embed_model = None
        self._llm = None
        self._vector_store = None
        self._index = None
    
    def _get_embed_model(self):
        """Get or create the embedding model (singleton per process)."""
        if self._embed_model is None:
            with self._lock:
                if self._embed_model is None:
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=EMBED_MODEL_NAME,
                        device=device,
                        trust_remote_code=True,
                        model_kwargs={"quantize": "static-int8"},
                    )
        return self._embed_model
    
    def _get_llm(self):
        """Get or create the LLM (singleton per process)."""
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    self._llm = LlamaCPP(
                        model_path=GGUF_MODEL_PATH,
                        temperature=0.3,
                        max_new_tokens=512,
                        context_window=2048,
                        model_kwargs={
                            "n_batch": N_BATCH,
                            "n_gpu_layers": N_GPU_LAYERS,
                        },
                        verbose=True,
                    )
        return self._llm
    
    def _get_vector_store(self):
        """Get or create the vector store (singleton per process)."""
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    self._vector_store = LanceDBVectorStore(
                        uri=DB_PATH, 
                        table_name=TABLE_NAME, 
                        mode="r"
                    )
        return self._vector_store
    
    def _get_index(self):
        """Get or create the index (singleton per process)."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    vector_store = self._get_vector_store()
                    storage_context = StorageContext.from_defaults(
                        persist_dir=os.path.join(DB_PATH, "li_storage"),
                        vector_store=vector_store,
                    )
                    self._index = load_index_from_storage(
                        storage_context, 
                        embed_model=self._get_embed_model()
                    )
        return self._index
    
    def create_query_engine(self, user_filters: MetadataFilters = None):
        """Create a new query engine instance with optional user filters."""
        # Get the shared components
        embed_model = self._get_embed_model()
        llm = self._get_llm()
        index = self._get_index()
        
        # Create a new retriever instance (not shared)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=BEAM_K,
            search_type="similarity",
        )
        
        # Apply user filters if provided
        if user_filters:
            retriever.vector_store_kwargs = {"filters": user_filters}
        
        # Create reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
            top_n=FINAL_K,
            device=device, 
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=PromptTemplate(QA_TEMPLATE),
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker],
            response_synthesizer=response_synthesizer,
        )

# Global factory instance
query_engine_factory = QueryEngineFactory()

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def process_query(query_text: str, user_id: str, group_ids: list):
    """Gradio wrapper – yields intermediate & final responses."""
    if not query_text:
        return "Please enter a question.", ""

    yield "Thinking...", ""

    # Create user-specific filters
    user_filter = ExactMatchFilter(key="user_id", value=user_id)
    group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
    
    # Combine filters: user can access their personal docs OR docs from their groups
    all_filters = [user_filter] + group_filters
    filters = MetadataFilters(filters=all_filters, condition="or")

    # Create a new query engine instance with user-specific filters
    query_engine = query_engine_factory.create_query_engine(user_filters=filters)

    response = query_engine.query(query_text)

    # Build a neat citation list
    sources_md = ""
    for sn in response.source_nodes:
        meta = sn.node.metadata
        doc = meta.get("document_name", "N/A")
        page = meta.get("page_number", "?")
        score = sn.score
        sources_md += f"- **Source:** {doc}, **Page:** {page} (Score: {score:.4f})\n"

    yield response.response, sources_md


def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Expert‑Call RAG Assistant\nAsk a question – the model fetches relevant transcript chunks, re‑ranks them, and answers with citations."
        )
        query_in = gr.Textbox(label="Your Question", lines=3)
        ask_btn = gr.Button("Submit")
        ans_md = gr.Markdown()
        src_md = gr.Markdown()

        ask_btn.click(process_query, inputs=query_in, outputs=[ans_md, src_md])

    demo.launch(share=False,inbrowser=True)


if __name__ == "__main__":
    main()
