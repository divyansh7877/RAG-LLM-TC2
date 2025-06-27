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
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./lancedb_expert_calls"
TABLE_NAME = "expert_call_embeddings"
EMBED_MODEL_NAME = "./models/gte-large-en-v1.5"  # INT8‐quantised, CPU
GGUF_MODEL_PATH = "./models/models--lmutimb--Meta-Llama-3.1-8B-Instruct-Q5_0-GGUF/snapshots/0bec7ad9d7a9cfbd8a6d2100f10e114bec3e44ad/meta-llama-3.1-8b-instruct-q5_0.gguf"
#GGUF_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"  # llama.cpp model\

has_cuda = torch.cuda.is_available()

# Runtime knobs
device = "cuda" if has_cuda else "cpu"  
BEAM_K = 30      # initial ANN beam for MMR
FINAL_K = 8     # chunks passed to the LLM
MMR_LAMBDA = 0.1
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
N_THREADS = mp.cpu_count()
N_GPU_LAYERS = -1 if has_cuda else 0 # set 0 if no GPU / VRAM < 12 GB
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
# Build the Retrieval‑Augmented QA engine
# ---------------------------------------------------------------------------

def build_query_engine():
    """Initialise embeddings, vector store, retriever, reranker, and LLM."""

    # 1⃣  Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device=device,
        trust_remote_code=True,
        model_kwargs={"quantize": "static-int8"},  # requires sentence-transformers >=3.3
    )


    # 2⃣  Llama‑3.1‑8B in llama.cpp with performance flags
    llm = LlamaCPP(
        #model_url = '', # for online models.
        model_path=GGUF_MODEL_PATH,
        temperature=0.3,
        max_new_tokens=512,
        context_window=2048,
        model_kwargs={
            #"n_threads": N_THREADS,
            "n_batch": N_BATCH,
            "n_gpu_layers": N_GPU_LAYERS,
            #"prompt_cache_path": PROMPT_CACHE_DIR,
            #"echo_prompt":True
            #"top_p":0.9,
        },
        verbose=True,
        
    )

    Settings.llm = llm

    # 3⃣  Vector store – open read‑only;
    vector_store = LanceDBVectorStore(uri=DB_PATH, table_name=TABLE_NAME, mode="r")
    storage_context = StorageContext.from_defaults(
        persist_dir=os.path.join(DB_PATH, "li_storage"),
        vector_store=vector_store,
    )
    index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)

    # 4⃣  Retriever – large beam + MMR diversity
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=BEAM_K,
        search_type="similarity",          # built‑in MMR search
        #search_kwargs={"lambda_mult": MMR_LAMBDA},
    )

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=FINAL_K,
        device=device, 
    )


    # 6⃣  Response synthesiser with custom citation prompt
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=PromptTemplate(QA_TEMPLATE),
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],  # Stage C
        response_synthesizer=response_synthesizer,
    )

# Build the engine once at startup
query_engine = build_query_engine()

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def process_query(query_text: str):
    """Gradio wrapper – yields intermediate & final responses."""
    if not query_text:
        yield "Please enter a question.", ""
        return

    yield "Thinking …", ""

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
