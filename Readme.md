# Project: Multi-User Private RAG System

## 1. Objective

This project provides a secure, multi-user, and private Retrieval-Augmented Generation (RAG) system. It allows different users to upload their own private PDF documents, which are then used as a knowledge base for a local Large Language Model (LLM). The system is designed to be modular, enabling easy customization of its core components like the LLM, embedding models, and retrieval strategies.

The key goals are:
- **Data Privacy:** Ensure users can only query their own documents or documents from groups they belong to.
- **Local First:** All components, including the LLM and vector database, run locally to prevent data from leaving the machine.
- **Accurate & Citable Answers:** The LLM is prompted to answer questions based *only* on the provided documents and to cite its sources by document name and page number.

---

## 2. Features

- **User Authentication:** A simple login system to manage access.
- **Multi-Tenant Data Storage:** Documents are associated with a `user_id` and `group_id`, allowing for both personal and shared knowledge bases.
- **PDF Document Upload:** Users can upload PDF files through a web interface.
- **End-to-End RAG Pipeline:**
    - **Ingestion:** Extracts text, splits it into chunks, and generates embeddings.
    - **Storage:** Stores embeddings and metadata in a LanceDB vector database.
    - **Retrieval:** Fetches relevant document chunks based on the user's query and access rights.
    - **Reranking:** Refines the retrieved results for better relevance.
    - **Generation:** Uses a local LLM to synthesize an answer from the retrieved context.
- **Web Interface:** A user-friendly UI built with Gradio for easy interaction.

---

## 3. System Architecture

The application is composed of three main Python scripts within the `app/` directory, orchestrated by a Gradio frontend.

### `app/main.py`: The Frontend and Orchestrator

- **Responsibilities:**
    - Renders the Gradio web UI.
    - Manages the user login flow and maintains user session state.
    - Handles the file upload interface, calling the embedding script as a subprocess.
    - Provides the query interface, passing user questions and credentials to the RAG engine.
- **User Management:** A simple dictionary `USERS` holds usernames, passwords, and group memberships.

### `app/new_embedder.py`: The Ingestion and Embedding Service

- **Trigger:** Called by `main.py` when a user uploads files.
- **Process:**
    1.  Receives a list of PDF file paths, along with the `user_id` and `group_id` of the owner.
    2.  Uses `PyMuPDF` to extract text from each page of the PDFs.
    3.  Cleans and splits the text into manageable chunks using `LlamaIndex`'s `SentenceSplitter`.
    4.  **Crucially, it attaches the `user_id` and `group_id` as metadata to each chunk.**
    5.  Uses the `gte-large-en-v1.5` sentence transformer to create vector embeddings for each chunk.
    6.  Connects to the `multi_user_db.lance` LanceDB database and stores the embeddings and their associated metadata.

### `app/new_rag_ui.py`: The Secure RAG Query Engine

- **Responsibilities:**
    - Initializes and holds the core components: the LLM, the embedding model, and the connection to the vector store.
    - Provides the `process_query` function that securely answers user questions.
- **Process:**
    1.  **Initialization (at startup):** Loads the `Llama-3.2-3B` GGUF model via `LlamaCPP` and the `gte-large-en-v1.5` embedding model. This is done only once to ensure good performance.
    2.  **Query Handling (per request):**
        a. Receives the query text, `user_id`, and `group_ids` from `main.py`.
        b. **Security Enforcement:** Creates metadata filters to search the vector database for chunks that match **either** the `user_id` (for personal documents) **or** one of the user's `group_ids`.
        c. **Retrieval:** Performs a vector search in LanceDB using these filters to find the most relevant document chunks the user is allowed to see.
        d. **Reranking:** Uses a `cross-encoder` model to re-rank the retrieved chunks for higher relevance.
        e. **Synthesis:** Passes the original question and the context from the reranked chunks to the LLM using a specific prompt template.
        f. **Citation:** The final answer includes citations to the source document and page number.

---

## 4. Core Components

-   **LLM:** `Llama-3.2-3B-Instruct-IQ3_M.gguf` (a quantized model for efficient local inference).
-   **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5` (quantized for CPU performance).
-   **Vector Database:** `LanceDB` (for efficient, file-based vector storage).
-   **Core Frameworks:** `LlamaIndex` (for the RAG pipeline), `Gradio` (for the UI).

---

## 5. Setup and Usage

### Installation

1.  It is recommended to use a virtual environment (e.g., `conda` or `venv`).
2.  Install the required Python packages using the provided files. For conda environments:
    ```bash
    conda env create -f environment.yml
    conda activate llm_rag
    ```
    Alternatively, using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  The `run.sh` script can be used to download the necessary models, but ensure the dependencies are installed first.

### Running the Application

To start the web interface, run the main application file:

```bash
python app/main.py
```

This will launch the Gradio server. Open the provided URL in your browser to access the application.

### How to Use

1.  **Login:** Use one of the credentials defined in `app/main.py`.
    -   e.g., Username: `assistant1`, Password: `password1`
2.  **Upload Documents:** Navigate to the "Upload" tab, select your PDF files, choose a destination ("Personal" or a shared group), and click "Upload".
3.  **Query Documents:** Navigate to the "Query" tab, type your question, and click "Submit". The model will generate an answer based on the documents you have access to.

---

## 6. Modularity and Customization

The system is designed to be modular:

-   **LLM:** To use a different model, change the `GGUF_MODEL_PATH` in `app/new_rag_ui.py` to point to another GGUF-compatible file.
-   **Retriever:** The retrieval logic in `app/new_rag_ui.py` can be modified to use different LlamaIndex retrievers (e.g., hybrid search, different MMR settings).
-   **Data Formats:** The embedding script (`app/new_embedder.py`) can be extended to support other file types (e.g., `.txt`, `.docx`) by adding new data loaders.

---

## 7. Known Issues & Limitations

**Concurrency:** The application in its current state is **NOT thread-safe** and will not handle concurrent users correctly.

-   **Embedding:** Concurrent embedding requests will work but are highly inefficient, as each request launches a separate process that loads the entire embedding model into memory. This will lead to very high CPU and RAM usage.
-   **Querying:** The query engine is **not thread-safe**. If multiple users query simultaneously, there is a race condition that can cause the security filters to be applied incorrectly, leading to **data leakage** where one user might see results from another user's private documents. **This is a critical issue that must be fixed before use in a production or multi-user environment.**