import gradio as gr
import os
import shutil
import subprocess
from functools import partial
import uuid
import time
import threading
from collections import defaultdict

# Import the new query function
from new_rag_ui import process_query
# Import embedding function directly
from new_embedder import embed_and_store, build_nodes_from_pdfs

# --- User Management (Prototype) ---
USERS = {
    "assistant1": {"password": "password1", "groups": ["assistance", "common_rules"]},
    "assistant2": {"password": "password2", "groups": ["assistance"]},
    "guest": {"password": "password", "groups": ["common_rules"]},
}

# --- Rate Limiting and Resource Management ---
class RateLimiter:
    """Thread-safe rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)  # user_id -> list of timestamps
        self._lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make a request."""
        current_time = time.time()
        
        with self._lock:
            # Clean old requests
            user_requests = self.requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests 
                              if current_time - req_time < self.window_seconds]
            
            # Check if under limit
            if len(user_requests) < self.max_requests:
                user_requests.append(current_time)
                return True
            
            return False
    
    def get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for a user."""
        current_time = time.time()
        
        with self._lock:
            user_requests = self.requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests 
                              if current_time - req_time < self.window_seconds]
            
            return max(0, self.max_requests - len(user_requests))

# Global rate limiters
query_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)  # 20 queries per minute
upload_rate_limiter = RateLimiter(max_requests=5, window_seconds=300)  # 5 uploads per 5 minutes

# --- Session Management ---
class SessionManager:
    """Thread-safe session manager for multi-user support."""
    
    def __init__(self):
        self._sessions = {}  # session_id -> user_data
        self._user_sessions = {}  # username -> session_id
        self._lock = threading.Lock()
    
    def create_session(self, username: str, groups: list) -> str:
        """Create a new session for a user."""
        with self._lock:
            # Remove any existing session for this user
            if username in self._user_sessions:
                old_session = self._user_sessions[username]
                if old_session in self._sessions:
                    del self._sessions[old_session]
            
            # Create new session
            session_id = str(uuid.uuid4())
            user_data = {
                "username": username,
                "groups": groups,
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            self._sessions[session_id] = user_data
            self._user_sessions[username] = session_id
            return session_id
    
    def get_session(self, session_id: str) -> dict:
        """Get session data by session ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["last_activity"] = time.time()
            return session
    
    def validate_session(self, session_id: str) -> bool:
        """Check if session is valid and not expired."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Session expires after 24 hours of inactivity
        if time.time() - session["last_activity"] > 86400:
            self.remove_session(session_id)
            return False
        
        return True
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        with self._lock:
            if session_id in self._sessions:
                username = self._sessions[session_id]["username"]
                if username in self._user_sessions:
                    del self._user_sessions[username]
                del self._sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if current_time - session["last_activity"] > 86400:  # 24 hours
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.remove_session(session_id)

# Global session manager
session_manager = SessionManager()

# --- Backend Functions ---

def embed_files(files, destination, session_id):
    """Embeds uploaded files directly without spawning subprocesses."""
    # Validate session
    user_data = session_manager.get_session(session_id)
    if not user_data:
        return "Error: Invalid or expired session. Please login again."
    
    username = user_data.get("username")
    if not username or not files or not destination:
        return "Error: Missing user info, files, or destination."

    # Check rate limiting
    if not upload_rate_limiter.is_allowed(username):
        remaining = upload_rate_limiter.get_remaining_requests(username)
        return f"Rate limit exceeded. You can upload {remaining} more files in the next 5 minutes."

    group_id = destination if destination != "Personal" else "personal"

    temp_dir = os.path.abspath("temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Copy files to temp directory
        filepaths = [shutil.copy(f.name, temp_dir) for f in files]
        
        # Build nodes with user/group metadata
        nodes = build_nodes_from_pdfs(
            filepaths,
            user_id=username,
            group_id=group_id,
            chunk_size=512,
            chunk_overlap=20
        )
        
        # Embed and store directly
        db_path = os.path.abspath("./multi_user_db.lance")
        embed_and_store(
            nodes,
            db_path=db_path,
            table_name="document_embeddings",
            embed_model_name="./models/gte-large-en-v1.5",
            device="cpu"
        )
        
        shutil.rmtree(temp_dir)
        return f"Successfully embedded {len(files)} files into '{destination}'."
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error embedding files: {str(e)}"

def query_wrapper(query, session_id):
    """Wrapper to pass user context to the query engine."""
    # Validate session
    user_data = session_manager.get_session(session_id)
    if not user_data:
        return "Error: Invalid or expired session. Please login again.", ""
    
    username = user_data.get("username")
    groups = user_data.get("groups", [])
    if not username:
        return "Error: User not logged in.", ""
    
    # Check rate limiting
    if not query_rate_limiter.is_allowed(username):
        remaining = query_rate_limiter.get_remaining_requests(username)
        return f"Rate limit exceeded. You can make {remaining} more queries in the next minute.", ""
    
    return process_query(query, username, groups)

# --- Main Application UI and Logic ---

def main():
    """Main function to run the Gradio app with a login screen."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        session_id_state = gr.State("")

        # --- Login View ---
        with gr.Column(visible=True) as login_view:
            gr.Markdown("## Please Login")
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            login_status = gr.Markdown()

        # --- Main App View (Initially Hidden) ---
        with gr.Column(visible=False) as main_app_view:
            welcome_markdown = gr.Markdown()
            
            with gr.Tab("Query"):
                query_input = gr.Textbox(label="Your Question", scale=4, container=False)
                submit_button = gr.Button("Submit")
                answer_output = gr.Markdown()
                sources_output = gr.Markdown()

            with gr.Tab("Upload"):
                file_uploader = gr.File(label="Upload PDF", file_count="multiple", type="filepath")
                destination_dropdown = gr.Dropdown(label="Destination", choices=["Personal"])
                upload_button = gr.Button("Upload")
                upload_status = gr.Markdown()

        # --- Event Handlers ---
        def do_login(username, password):
            if username in USERS and USERS[username]["password"] == password:
                # Create session
                session_id = session_manager.create_session(username, USERS[username]["groups"])
                upload_opts = ["Personal"] + USERS[username]["groups"]
                return {
                    session_id_state: session_id,
                    login_view: gr.update(visible=False),
                    main_app_view: gr.update(visible=True),
                    welcome_markdown: gr.update(value=f"# Welcome, {username}!"),
                    destination_dropdown: gr.update(choices=upload_opts, value="Personal"),
                    login_status: gr.update(value=""),
                }
            else:
                return {login_status: gr.update(value="Invalid username or password.")}

        login_button.click(
            do_login, 
            [username_input, password_input], 
            [session_id_state, login_view, main_app_view, welcome_markdown, destination_dropdown, login_status]
        )

        submit_button.click(
            query_wrapper, 
            inputs=[query_input, session_id_state], 
            outputs=[answer_output, sources_output]
        )

        upload_button.click(
            embed_files, 
            inputs=[file_uploader, destination_dropdown, session_id_state], 
            outputs=[upload_status]
        )

    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    main()
