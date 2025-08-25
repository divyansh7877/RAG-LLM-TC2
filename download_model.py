from huggingface_hub import hf_hub_download, snapshot_download

# Download the LLM
print("Downloading LLM...")
llm_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-IQ3_M.gguf",
    local_dir="./models/",
    local_dir_use_symlinks=False
)


llm_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    local_dir="./models/",
    local_dir_use_symlinks=False
)

print(f"LLM downloaded to: {llm_path}")

# Download the Embedding Model
print("\nDownloading Embedding Model...")
embedding_path = snapshot_download(
    repo_id="Alibaba-NLP/gte-large-en-v1.5",
    local_dir="./models/gte-large-en-v1.5",
    local_dir_use_symlinks=False
)
print(f"Embedding model downloaded to: {embedding_path}")

# Download the Reranker (optional; used if ENABLE_RERANKER=1)
print("\nDownloading Reranker model (cross-encoder/ms-marco-MiniLM-L6-v2)...")
reranker_path = snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    local_dir="./models/cross-encoder/ms-marco-MiniLM-L6-v2",
    local_dir_use_symlinks=False
)
print(f"Reranker downloaded to: {reranker_path}")