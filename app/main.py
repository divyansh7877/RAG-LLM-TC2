import gradio as gr
import os
import shutil
import subprocess
from functools import partial

# Import the new query function
from new_rag_ui import process_query

# --- User Management (Prototype) ---
USERS = {
    "assistant1": {"password": "password1", "groups": ["assistance", "common_rules"]},
    "assistant2": {"password": "password2", "groups": ["assistance"]},
    "guest": {"password": "password", "groups": ["common_rules"]},
}

# --- Backend Functions ---

def embed_files(files, destination, user_info):
    """Calls the new_embedder.py script to embed uploaded files."""
    username = user_info.get("username")
    if not username or not files or not destination:
        return "Error: Missing user info, files, or destination."

    group_id = destination if destination != "Personal" else "personal"

    temp_dir = os.path.abspath("temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    filepaths = [shutil.copy(f.name, temp_dir) for f in files]

    embedder_script_path = os.path.abspath("app/new_embedder.py")
    db_path = os.path.abspath("./multi_user_db.lance")
    table_name = "document_embeddings"
    
    command = [
        "python", embedder_script_path,
        f"--user_id={username}", f"--group_id={group_id}",
        f"--db={db_path}", f"--table={table_name}", "--device=cpu",
    ] + filepaths

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        shutil.rmtree(temp_dir)
        return f"Successfully embedded {len(files)} files into '{destination}'."
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        return f"Error embedding files: {e.stderr}"

def query_wrapper(query, user_info):
    """Wrapper to pass user context to the query engine."""
    username = user_info.get("username")
    groups = user_info.get("groups", [])
    if not username:
        return "Error: User not logged in.", ""
    return process_query(query, username, groups)

# --- Main Application UI and Logic ---

def main():
    """Main function to run the Gradio app with a login screen."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        user_info_state = gr.State({})

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
                user_groups = USERS.get(username_input.value, {}).get("groups", [])
                upload_options = ["Personal"] + user_groups
                
                file_uploader = gr.File(label="Upload PDF", file_count="multiple", type="filepath")
                destination_dropdown = gr.Dropdown(label="Destination", choices=upload_options)
                upload_button = gr.Button("Upload")
                upload_status = gr.Markdown()

        # --- Event Handlers ---
        def do_login(username, password):
            if username in USERS and USERS[username]["password"] == password:
                user_data = {"username": username, "groups": USERS[username]["groups"]}
                upload_opts = ["Personal"] + user_data["groups"]
                return {
                    user_info_state: user_data,
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
            [user_info_state, login_view, main_app_view, welcome_markdown, destination_dropdown, login_status]
        )

        submit_button.click(
            query_wrapper, 
            inputs=[query_input, user_info_state], 
            outputs=[answer_output, sources_output]
        )

        upload_button.click(
            embed_files, 
            inputs=[file_uploader, destination_dropdown, user_info_state], 
            outputs=[upload_status]
        )

    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    main()
