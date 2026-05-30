#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

__author__ = "Gabriel Zapodeanu, Principal TME"
__email__ = "gzapodea@cisco.com"
__version__ = "0.1.0"
__copyright__ = "Copyright (c) 2026 Cisco and/or its affiliates."
__license__ = "Cisco Sample Code License, Version 1.1"

import os
import re
import difflib
import time

import certifi
import chromadb
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, "environment.env")
load_dotenv(ENV_PATH)

# Database server details
DB_SERVER = os.getenv("DB_SERVER")
DB_PORT = os.getenv("DB_PORT")
DB_COLLECTION = os.getenv("DB_COLLECTION")

# Anthropic config
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")

# Embeddings config
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH")
HF_CA_BUNDLE = os.getenv("HF_CA_BUNDLE")
DATASET = os.getenv("DATASET")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dataset_path_from_env():
    """Resolve DATASET path relative to repo root when needed."""
    if not DATASET:
        return None
    if os.path.isabs(DATASET):
        return DATASET
    return os.path.join(BASE_DIR, DATASET)


def known_devices_from_dataset():
    """Derive known device names from DATASET filenames."""
    path = dataset_path_from_env()
    if not path or not os.path.isdir(path):
        return []
    devices = set()
    for filename in os.listdir(path):
        if "_" in filename:
            devices.add(filename.split("_", 1)[0].strip())
    return sorted(d for d in devices if d)


def normalize_for_match(value):
    return value.lower().replace("-", " ").replace("_", " ").strip()


def detect_device(query, known_devices):
    normalized_query = normalize_for_match(query)
    for device in known_devices:
        if normalize_for_match(device) in normalized_query:
            return device
    return None


def detect_device_fuzzy(query, known_devices, cutoff=0.74):
    if not known_devices:
        return None, None
    normalized_to_original = {normalize_for_match(item): item for item in known_devices}
    tokens = re.findall(r"[A-Za-z0-9_-]+", query)
    for token in tokens:
        normalized_token = normalize_for_match(token)
        if not normalized_token:
            continue
        match = difflib.get_close_matches(
            normalized_token,
            list(normalized_to_original.keys()),
            n=1,
            cutoff=cutoff,
        )
        if match:
            return normalized_to_original[match[0]], token
    return None, None


def resolve_device_and_query(query, known_devices):
    notices = []
    updated_query = query
    device = detect_device(query, known_devices)
    if not device:
        fuzzy_device, typed_value = detect_device_fuzzy(query, known_devices)
        if fuzzy_device:
            device = fuzzy_device
            notices.append(f"NetworkInsights: Interpreting '{typed_value}' as '{fuzzy_device}'.")
            updated_query = re.sub(rf"\b{re.escape(typed_value)}\b", fuzzy_device, query, flags=re.IGNORECASE)
    return device, updated_query, notices


def build_metadata_filter(device):
    if not device:
        return None
    return {"device name": device}


def tokenize_text(value):
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}


def rerank_docs_by_query_overlap(query, docs):
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return docs
    scored = []
    for idx, doc in enumerate(docs):
        content_tokens = tokenize_text(doc.page_content)
        overlap = len(query_tokens.intersection(content_tokens))
        metadata_text = " ".join(
            str(doc.metadata.get(key, "")) for key in ("CLI command", "filename")
        )
        metadata_tokens = tokenize_text(metadata_text)
        metadata_overlap = len(query_tokens.intersection(metadata_tokens))
        score = overlap + (0.5 * metadata_overlap)
        scored.append((score, idx, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def fetch_filter_candidates(chroma_db, metadata_filter, limit=200):
    if not metadata_filter:
        return []
    try:
        payload = chroma_db.get(where=metadata_filter, include=["documents", "metadatas"], limit=limit)
    except Exception:
        return []
    docs = []
    for content, metadata in zip(payload.get("documents", []), payload.get("metadatas", [])):
        docs.append(Document(page_content=content or "", metadata=metadata or {}))
    return docs


def merge_unique_docs(primary_docs, extra_docs):
    merged = []
    seen = set()
    for doc in primary_docs + extra_docs:
        key = (doc.page_content, tuple(sorted((doc.metadata or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def format_docs(docs):
    """Format retrieved docs into a prompt-friendly context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def validate_runtime_env():
    required = {
        "DB_SERVER": DB_SERVER,
        "DB_PORT": DB_PORT,
        "DB_COLLECTION": DB_COLLECTION,
        "CLAUDE_API_KEY": CLAUDE_API_KEY,
        "CLAUDE_MODEL": CLAUDE_MODEL,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    if not (MODEL_LOCAL_PATH or MODEL_NAME):
        raise ValueError("Missing required embedding model configuration: set MODEL_LOCAL_PATH or MODEL_NAME")
    try:
        db_port = int(DB_PORT)
    except ValueError as exc:
        raise ValueError("DB_PORT must be an integer value") from exc
    return db_port, (MODEL_LOCAL_PATH if MODEL_LOCAL_PATH else MODEL_NAME)


def print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms):
    print(
        f"NetworkInsights: timing retrieval={retrieval_ms:.0f}ms "
        f"filter={filter_ms:.0f}ms rerank={rerank_ms:.0f}ms "
        f"llm={llm_ms:.0f}ms total={total_ms:.0f}ms"
    )


def main():
    """Conversational Anthropic RAG app using LCEL."""

    # TLS trust settings for HuggingFace downloads.
    if HF_CA_BUNDLE:
        os.environ["SSL_CERT_FILE"] = HF_CA_BUNDLE
        os.environ["REQUESTS_CA_BUNDLE"] = HF_CA_BUNDLE
    else:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    try:
        db_port, effective_model_name = validate_runtime_env()
    except ValueError as exc:
        print(f"NetworkInsights: {exc}")
        return

    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=db_port)
    embeddings = HuggingFaceEmbeddings(model_name=effective_model_name)
    chroma_db = Chroma(
        client=chroma_db_server,
        collection_name=DB_COLLECTION,
        embedding_function=embeddings,
    )
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=CLAUDE_API_KEY,
        temperature=1,
        max_tokens=1024,
    )

    print(f"Target Chroma server: {DB_SERVER}:{DB_PORT}")
    print(f"Target collection: {DB_COLLECTION}")
    print(f"Anthropic model: {CLAUDE_MODEL}")
    print(f"Embedding model: {effective_model_name}")

    known_devices = known_devices_from_dataset()
    if known_devices:
        print(f"Known devices in dataset: {', '.join(known_devices)}")

    genaiops_prompt = (
        "You are an assistant for network insights tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "Consider the conversation history when answering follow-up questions. "
        "If the user refers to previous topics (like 'that', 'it', 'those steps'), "
        "use the chat history to understand what they're referring to. "
        "If you don't know the answer, say that you don't know. "
        "The user is networking knowledgeable.\n\n"
        "Retrieved Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", genaiops_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    print("\nHi, I am NetworkInsights. Ask a question about your network data, or press Enter to exit.\n")
    chat_history = []

    while True:
        query = input("Your input: ").strip()
        if not query:
            print("\nNetworkInsights. Goodbye!\n")
            break

        query_start = time.perf_counter()
        retrieval_ms = 0.0
        filter_ms = 0.0
        rerank_ms = 0.0
        llm_ms = 0.0

        detected_device, effective_query, notices = resolve_device_and_query(query, known_devices)
        search_kwargs = {"k": 16}
        metadata_filter = build_metadata_filter(detected_device)
        for notice in notices:
            print(notice)
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
            print(f"NetworkInsights: Applying metadata filter {metadata_filter}")

        search_kwargs["k"] = 20
        retriever = chroma_db.as_retriever(search_kwargs=search_kwargs)
        retrieval_start = time.perf_counter()
        retrieved_docs = retriever.invoke(effective_query)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        filter_start = time.perf_counter()
        filter_candidates = fetch_filter_candidates(
            chroma_db, metadata_filter, limit=max(80, search_kwargs["k"] * 4)
        )
        filter_ms = (time.perf_counter() - filter_start) * 1000

        rerank_start = time.perf_counter()
        candidate_docs = merge_unique_docs(retrieved_docs, filter_candidates)
        matching_docs = rerank_docs_by_query_overlap(effective_query, candidate_docs)[:16]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        if not matching_docs:
            print("NetworkInsights: No matching context found for this query.\n")
            total_ms = (time.perf_counter() - query_start) * 1000
            print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms)
            continue

        answer_chain = (
            {
                "context": RunnableLambda(lambda _: format_docs(matching_docs)),
                "input": RunnableLambda(lambda _: effective_query),
                "chat_history": RunnableLambda(lambda _: chat_history),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        llm_start = time.perf_counter()
        answer = answer_chain.invoke({})
        llm_ms = (time.perf_counter() - llm_start) * 1000
        if not answer:
            print("NetworkInsights: No answer generated for this query.\n")
            total_ms = (time.perf_counter() - query_start) * 1000
            print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms)
            continue

        print("NetworkInsights: " + answer + "\n")
        total_ms = (time.perf_counter() - query_start) * 1000
        print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms)

        chat_history.extend(
            [
                HumanMessage(content=effective_query),
                AIMessage(content=answer),
            ]
        )
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


if __name__ == "__main__":
    main()
