import streamlit as st
import requests
import uuid
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="RAG HR CHATBOT",
    page_icon="🤖",
    layout="wide"
)

# CSS for modern fonts and design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;500;700&display=swap');

body, .stApp {background-color:#0f0f23; color:white; font-family:'Rajdhani', sans-serif;}
h1,h2,h3,h4,h5,h6 {font-family:'Orbitron', sans-serif; color:#00d4ff; margin:0;}
.stButton>button {background-color:#764ba2; color:white; border-radius:10px; padding:0.5rem 1rem; width:100%; cursor:pointer; font-family:'Rajdhani', sans-serif;}
.stChatMessage>div {border-radius:15px; padding:1rem; margin-bottom:0.8rem; font-family:'Rajdhani', sans-serif;}
.stChatMessage[data-testid*="user"] > div {background:#ff6b35; color:white; margin-left:auto;}
.stChatMessage[data-testid*="assistant"] > div {background:#16213e; color:#00f5ff; margin-right:auto;}
.stChatInput input {background:#1a2332; color:white; border-radius:20px; padding:0.5rem 1rem; font-family:'Rajdhani', sans-serif;}
::-webkit-scrollbar {width:8px;}
::-webkit-scrollbar-thumb {background:#764ba2; border-radius:4px;}
.sidebar .sidebar-content {background:#16213e;}
footer {text-align:center; color:#94a3b8; padding:0.5rem;}
</style>
""", unsafe_allow_html=True)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper functions
def get_backend_status():
    try:
        requests.get("http://localhost:5000", timeout=2)
        return "ONLINE"
    except:
        return "OFFLINE"

def new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.stop()  # refresh app

# Header
st.markdown(f"""
<div style="padding:1rem; background:#16213e; border-bottom:2px solid #00f5ff; border-radius:5px; margin-bottom:1rem;">
    <h1>RAG HR CHATBOT</h1>
    <h3 style="color:#7c3aed;">Your AI-powered HR Assistant</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🚀 HR Assistant")
    st.button("New Chat", on_click=new_chat)
    st.markdown("---")
    
    # Backend Status button
    if st.button("Check Backend Status"):
        status = get_backend_status()
        if status == "ONLINE":
            st.success("✅ Backend is ONLINE")
        else:
            st.error("❌ Backend is OFFLINE")
    
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    st.markdown("- Remote work policy\n- Leave policy\n- Benefits info\n- Compliance guidance")
    
    st.markdown("---")
    st.markdown("Company: **Datamites**")
    st.markdown("Made by **Likith**")

# Chat display
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
else:
    st.markdown("""
    <div style="text-align:center; margin-top:5rem;">
        <h3 style="color:#00f5ff;">NEXUS READY</h3>
        <p style="color:#94a3b8;">Your advanced HR assistant is online. Ask anything about policies, benefits, procedures, or compliance.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask your HR Assistant...")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Processing... ⏳")
    
    # Backend call
    try:
        resp = requests.post(
            "http://localhost:5000/query", 
            json={"query": query, "session_id": st.session_state.session_id}
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "No answer provided.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            placeholder.markdown(answer)
        else:
            placeholder.markdown("⚠️ System error. Try again.")
    except:
        placeholder.markdown("🔌 Backend unavailable.")
