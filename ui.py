import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage, SystemMessage
from main import get_rag_chain

st.set_page_config(page_title="Placement Assistant", layout="wide")

# Load default API key from .env
load_dotenv()
default_gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Step 0: Ask for Gemini API Key + Username if not already set
if "username" not in st.session_state:
    st.markdown("<h1 style='text-align: center;'>🎓 Placement Assistant Chatbot</h1>", unsafe_allow_html=True)
    username = st.text_input("Enter your name to start chatting 👇", key="name_input")
    gemini_api_key = st.text_input("Enter your Google Gemini API key (leave blank to use default):", type="password")
    st.write("Note: If you want to use the default api then the you may not get the proper result.")

    if username and st.button("Start using"):
        st.session_state.username = username
        st.session_state.gemini_api_key = gemini_api_key.strip() if gemini_api_key else default_gemini_api_key
        st.session_state.model_name = "gemini-1.5-flash" if gemini_api_key else "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key  # Apply to env for any backend usage
        st.rerun()
    st.stop()

# === CSS for custom layout ===
st.markdown("""
<style>
.header {
    position: fixed;
    top: 0;
    left: 0;
    padding-top: 1rem;
    background-color: white;
    z-index: 999;
    padding: 0.8rem;
    border-bottom: 1px solid #eee;
    font-size: 24px;
    font-weight: 600;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# Step 2: Chat interface
st.markdown(f"<h2 class='header'>🎓 Welcome to Placement Assistant Chatbot, {st.session_state.username}!</h2>", unsafe_allow_html=True)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# Show chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        # Initial of the username as avatar
        initial = st.session_state.username[0].upper()
        message(msg["content"], is_user=True, avatar_style="initials", seed=initial, key=f"user_{i}")
    else:
        message(msg["content"], is_user=False, avatar_style="initials", seed="AI", key=f"bot_{i}")



# Step 1: Accept input
user_input = st.chat_input(
        f"Ask me anything about placements, {st.session_state.get('username', 'User')}...",
        accept_file=True,
        file_type=["pdf"]
)
#user = chat_input_widget()
#with bottom():

model_name = st.session_state.get("model_name", "gemini-2.0-flash")
api_key = st.session_state.get("gemini_api_key", os.getenv("GOOGLE_API_KEY"))

rag_chain = get_rag_chain(model_name=model_name, api_key=api_key)

# Step 2: Capture input and immediately display user message
if user_input:
    text = user_input.text
    files = user_input.files
    resume_text = ""

    if files:
        pdf_file = files[0]
        reader = PdfReader(pdf_file)
        resume_text = "".join(page.extract_text() or "" for page in reader.pages)

    # Build input_text
    if text and resume_text:
        input_text = f"Resume: {resume_text}\n\nBased on the given resume, answer this query:\n{text}"
        display_text = f"Uploaded {pdf_file.name}\n\n{text}"
    elif resume_text:
        input_text = f"Resume: {resume_text}\n\nBased on the resume, suggest companies that are eligible."
        display_text = f"Uploaded {pdf_file.name}"
    else:
        input_text = text
        display_text = text

    # Store user message and processing instruction
    st.session_state.messages.append({"role": "user", "content": display_text})
    st.session_state.pending_input = input_text
    st.rerun()

# Step 3: Process pending input AFTER message is shown
if st.session_state.pending_input:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({
            "input": st.session_state.pending_input,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append(HumanMessage(content=st.session_state.pending_input))
        st.session_state.chat_history.append(SystemMessage(content=result["answer"]))
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.pending_input = None
        st.rerun()


