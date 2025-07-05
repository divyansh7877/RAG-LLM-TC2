#!/usr/bin/env python3
"""
Embed PDF transcripts into LanceDB. 

Usage
-----
```
python embedding_with_metadata_optimized.py ./pdfs/*.pdf --db ./lancedb_expert_calls --device cpu
```

Requirements (pip install)
--------------------------
* pymupdf
* lancedb
* llama‑index>=0.10.35
* sentence‑transformers>=3.3
* torch
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import List, Tuple

import fitz  # PyMuPDF
import lancedb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# ────────────────────────── PDF helpers ────────────────────────────

def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """Return a list of (page_text, page_number) tuples (1‑indexed)."""
    doc = fitz.open(pdf_path)
    pages: List[Tuple[str, int]] = []
    for i in range(len(doc)):
        pages.append((doc.load_page(i).get_text(), i + 1))
    doc.close()
    return pages


def clean_text(raw_text: str) -> str:
    """Basic cleanup: trim spaces and collapse excess newlines/spaces."""
    txt = re.sub(r"\n{3,}", "\n\n", raw_text)
    txt = "\n".join(line.strip() for line in txt.split("\n"))
    txt = re.sub(r" {2,}", " ", txt)
    return txt.strip()

# ────────────────────────── Node creation ──────────────────────────

def nodes_from_single_pdf(
    pdf_path: str,
    user_id: str,
    group_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
):
    """Create Llama‑Index Nodes with doc/page metadata from one PDF."""
    document_name = os.path.basename(pdf_path)
    pages = extract_pages_from_pdf(pdf_path)

    documents: List[Document] = []
    for text, page_number in pages:
        cleaned = clean_text(text)
        metadata = {
            "document_name": document_name,
            "page_number": page_number,
            "user_id": user_id,
            "group_id": group_id,
        }
        documents.append(
            Document(
                text=cleaned,
                metadata=metadata,
                id_=f"{document_name}_p{page_number}",
            )
        )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,  # propagate metadata into every node
    )
    return splitter.get_nodes_from_documents(documents)


def build_nodes_from_pdfs(
    pdf_paths: List[str],
    user_id: str,
    group_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
):
    """Aggregate nodes from a list of PDFs."""
    nodes = []
    for path in pdf_paths:
        nodes.extend(
            nodes_from_single_pdf(
                path,
                user_id=user_id,
                group_id=group_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return nodes

# ────────────────────────── Ingestion & storage ────────────────────

def embed_and_store(
    nodes,
    db_path: str,
    table_name: str = "expert_call_embeddings",
    embed_model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
    device: str = "cpu",
):
    """Embed **once**, store vectors + metadata in LanceDB."""

    # Initialise quantised embedding model for LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device,
        trust_remote_code=True,
        model_kwargs={"quantize": "static-int8"},
    )

    # Connect / create LanceDB store
    ldb = lancedb.connect(db_path)
    vector_store = LanceDBVectorStore(uri=db_path, table_name=table_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build or append to index; embeddings computed lazily per node
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    # Persist on disk alongside the LanceDB folder (for Llama‑Index metadata)
    persist_dir = os.path.join(db_path, "li_storage")
    index.storage_context.persist(persist_dir)

    # Quick sanity check
    tbl = ldb.open_table(table_name)
    print(
        f"\n✅ Stored {tbl.count_rows()} vectors in '{table_name}'. "
        "Each row contains 'document_name' and 'page_number' metadata."
    )

# ────────────────────────── CLI entry ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Embeds PDF documents into a LanceDB table with user/group metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pdfs", nargs="+", help="Path(s) or glob pattern(s) for PDF files.")
    parser.add_argument("--db", default="./multi_user_db.lance", help="Path to the LanceDB database directory.")
    parser.add_argument("--table", default="document_embeddings", help="Name of the table to create or use.")
    parser.add_argument("--user_id", required=True, help="The user ID to associate with these documents.")
    parser.add_argument("--group_id", required=True, help="The group ID to associate with these documents (e.g., 'personal', 'assistance').")
    parser.add_argument("--model_dir", default="./models/gte-large-en-v1.5", help="Path to the local HuggingFace embedding model.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use for embedding ('cpu' or 'cuda').")
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of text chunks for splitting.")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Overlap between text chunks.")
    args = parser.parse_args()

    # Expand potential globs (e.g. ./pdfs/*.pdf)
    pdf_paths: List[str] = []
    for pattern in args.pdfs:
        pdf_paths.extend(glob.glob(pattern))
    pdf_paths = sorted(set(pdf_paths))
    if not pdf_paths:
        raise SystemExit(f"No PDF files matched the given pattern(s): {args.pdfs}")

    print(f"🔍 Found {len(pdf_paths)} PDFs. Generating nodes for user '{args.user_id}' in group '{args.group_id}'…")
    nodes = build_nodes_from_pdfs(
        pdf_paths,
        user_id=args.user_id,
        group_id=args.group_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    print(f"🧩 Created {len(nodes)} nodes. Starting embedding → LanceDB …")

    embed_and_store(
        nodes,
        db_path=args.db,
        table_name=args.table,
        device=args.device,
        embed_model_name=args.model_dir
    )


if __name__ == "__main__":
    main()
