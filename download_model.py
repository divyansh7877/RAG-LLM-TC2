from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-IQ3_M.gguf",
    local_dir="./models/",       # exact folder you want the file in
)
print(file_path)  # /opt/llama/gguf/Llama-3.2-3B-Instruct-IQ3_M.gguf
