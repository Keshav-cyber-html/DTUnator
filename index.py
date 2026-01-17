import os
import sys
import time
import json
import logging
import datetime
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
import streamlit as st
import weaviate
from weaviate.embedded import EmbeddedOptions
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ONDEMAND_API_KEY = os.getenv("ONDEMAND_API_KEY")
if not ONDEMAND_API_KEY:
    logger.warning("ONDEMAND_API_KEY missing")

DB_PATH = "./vector_db"
INDEX_NAME = "DtuDoubt"

ONDEMAND_URL = "https://api.on-demand.io/chat/v1/sessions/query"
ONDEMAND_ENDPOINT_ID = "predefined-openai-gpt4.1-nano"

MAX_HISTORY_TURNS = 4


@dataclass(frozen=True)
class AppConfig:
    PAGE_TITLE: str = "DTUnator"
    PAGE_ICON: str = "⚡"
    RATE_LIMIT_SEC: float = 1.0


@st.cache_resource
def load_models():
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info("Loading cross-encoder...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return embeddings, cross_encoder


embeddings_model, cross_encoder_model = load_models()


def get_weaviate_client():
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(persistence_data_path=DB_PATH)
    )
    client.connect()
    return client


def retrieve_docs(query, k=12):
    client = get_weaviate_client()
    docs = []
    try:
        query_vector = embeddings_model.embed_query(query)
        result = client.collections.get(INDEX_NAME).query.near_vector(
            near_vector=query_vector, limit=k, return_properties=["text", "source"]
        )

        for obj in result.objects:
            text = obj.properties.get("text")
            if isinstance(text, str) and text.strip():
                docs.append(
                    {
                        "content": text,
                        "metadata": {"source": obj.properties.get("source", "Unknown")},
                    }
                )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
    finally:
        client.close()

    return docs


def cross_rerank(query, docs, top_k=5):
    if not docs:
        return []

    pairs = [(query, d["content"]) for d in docs]
    scores = cross_encoder_model.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [
        {
            "rank": i + 1,
            "score": float(score),
            "content": doc["content"],
            "metadata": doc["metadata"],
        }
        for i, (doc, score) in enumerate(ranked[:top_k])
    ]


def format_history(messages):
    history = messages[-MAX_HISTORY_TURNS * 2 :]
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm_api(prompt, system_instruction):
    payload = {
        "endpointId": ONDEMAND_ENDPOINT_ID,
        "responseMode": "sync",
        "query": f"{system_instruction}\n\n{prompt}",
    }

    headers = {"apikey": ONDEMAND_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            ONDEMAND_URL, json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        return response.json()["data"]["answer"]
    except Exception as e:
        logger.error(e)
        return "Sorry, something went wrong while answering."


def generate_rag_response(user_query, history):
    retrieved = retrieve_docs(user_query)
    reranked = cross_rerank(user_query, retrieved)

    history_block = format_history(history)

    strong_context = bool(reranked) and reranked[0]["score"] > 6.5

    if strong_context:
        context_text = "\n\n".join(d["content"] for d in reranked)

        prompt = f"""
Conversation so far:
{history_block}

Lecture Notes:
{context_text}

Question:
{user_query}
"""

        system_instruction = (
            "You are DTUnator, a helpful DTU study assistant.\n"
            "Use the lecture notes as your primary source.\n"
            "You are intellectually cursious and humble"
            "Have a bubbly personality"
            "Fix the font, write the reponse in a next line after the user query"
            "Ask permision to search on prior knowledge if neccessary.\n"
            "Be accurate. If unsure, say so.\n"
            "Do NOT fabricate facts.\n"
            "Ask before using Prior Information\n"
        )
    else:
        prompt = f"""
Conversation so far:
{history_block}

User question:
{user_query}
"""

        system_instruction = (
            "You are DTUnator, a helpful DTU study assistant.\n"
            "Use the lecture notes as your primary source.\n"
            "You are intellectually cursious and humble"
            "Have a bubbly personality"
            "Fix the font, write the reponse in a next line after the user query"
            "Ask permision to search on prior knowledge if neccessary.\n"
            "Be accurate. If unsure, say so.\n"
            "Do NOT fabricate facts.\n"
            "Ask before using Prior Information\n"
        )

    return call_llm_api(prompt, system_instruction)


class LLMService:
    def generate_response(self, question: str, history: list):
        response = generate_rag_response(question, history)

        for word in response.split():
            yield word + " "
            time.sleep(0.015)


class AssistantUI:
    def __init__(self, service: LLMService, config: AppConfig):
        self.service = service
        self.config = config
        self._init_state()

    def _init_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_request_time" not in st.session_state:
            st.session_state.last_request_time = datetime.datetime.min

    def render_header(self):
        st.markdown(
            f"""
            <h1>{self.config.PAGE_ICON} {self.config.PAGE_TITLE}</h1>
            <p style="color:gray;">Your AI companion for DTU lecture notes</p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    def render_chat(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask something..."):
            self.process_input(user_input)

    def process_input(self, user_input):
        now = datetime.datetime.now()
        if (
            now - st.session_state.last_request_time
        ).total_seconds() < self.config.RATE_LIMIT_SEC:
            st.toast("Slow down a bit ⏳")
            return

        st.session_state.last_request_time = now
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                stream = self.service.generate_response(
                    user_input, st.session_state.messages
                )
                full_response = st.write_stream(stream)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()


def main():
    st.set_page_config(page_title="DTUnator", page_icon="⚡", layout="wide")

    app = AssistantUI(service=LLMService(), config=AppConfig())

    app.render_header()
    app.render_chat()


main()
