"""Minimal chat UI for the RAG Support API.

A thin client: all the real logic (retrieval, generation, persistence)
lives in the FastAPI service. This just calls POST /chat and renders the
answer plus its cited sources -- run it with:

    streamlit run frontend/streamlit_app.py

against a running `rag-support serve` instance (defaults to
http://localhost:8000, overridable in the sidebar or via API_BASE_URL).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Northwind Cloud Support", page_icon="\U0001f4ac", layout="centered")

DEFAULT_API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content, sources)]

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", value=DEFAULT_API_BASE)
    category = st.selectbox(
        "Restrict to category",
        ["(any)", "ACCOUNT", "BILLING", "ORDERS", "SHIPPING", "RETURNS", "TECHNICAL",
         "SUBSCRIPTION", "PRIVACY", "CONTACT"],
    )
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    try:
        health = requests.get(f"{api_base}/health", timeout=5).json()
        st.success("API reachable")
        st.json(health)
    except requests.RequestException as exc:
        st.error(f"API not reachable: {exc}")

st.title("Northwind Cloud Support")
st.caption("Ask a question about your account, billing, orders, or the product. "
           "Answers are grounded in the support knowledge base -- sources are shown below each reply.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['rank']}] {source['title'] or source['category']}** "
                        f"· {source['category']} · similarity {source['similarity']:.2f}\n\n"
                        f"> {source['excerpt']}"
                    )

question = st.chat_input("How can we help?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "question": question,
        "conversation_id": st.session_state.conversation_id,
        "category": None if category == "(any)" else category,
    }
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                response = requests.post(f"{api_base}/chat", json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            st.session_state.conversation_id = data["conversation_id"]
            st.markdown(data["answer"])
            if data["sources"]:
                with st.expander(f"Sources ({len(data['sources'])})"):
                    for source in data["sources"]:
                        st.markdown(
                            f"**[{source['rank']}] {source['title'] or source['category']}** "
                            f"· {source['category']} · similarity {source['similarity']:.2f}\n\n"
                            f"> {source['excerpt']}"
                        )
            st.caption(f"embedding: {data['embedding_provider']} · llm: {data['llm_provider']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
            )
