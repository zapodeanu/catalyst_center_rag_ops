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
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, 'environment.env')
load_dotenv(ENV_PATH)

# database server details
DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = os.getenv('DB_PORT')
DB_COLLECTION = os.getenv('DB_COLLECTION')

# OpenAI key and model
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

# Embeddings model
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH')
HF_CA_BUNDLE = os.getenv('HF_CA_BUNDLE')
DATASET = os.getenv('DATASET')

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


def validate_runtime_env():
    required = {
        "DB_SERVER": DB_SERVER,
        "DB_PORT": DB_PORT,
        "DB_COLLECTION": DB_COLLECTION,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
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
    """
    A query app for network insights, powered by LangChain. Retrieves similarity matches
    from Chroma and generates responses using OpenAI's GPT-5.5.
    """

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

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Chroma DB server details and connection
    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=db_port)

    # Define the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=effective_model_name)

    # Chroma DB connection to server and collection
    chroma_db = Chroma(
        client=chroma_db_server,
        collection_name=DB_COLLECTION,
        embedding_function=embeddings
    )

    # Define the LLM used - OpenAI model from environment.env
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=1)

    print(f"Target Chroma server: {DB_SERVER}:{DB_PORT}")
    print(f"Target collection: {DB_COLLECTION}")
    print(f"OpenAI model: {OPENAI_MODEL}")
    print(f"Embedding model: {effective_model_name}")
    known_devices = known_devices_from_dataset()
    if known_devices:
        print(f"Known devices in dataset: {', '.join(known_devices)}")

    # Create the prompt
    genaiops_prompt = (
        "You are an assistant for network insights tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. The user is networking knowledgeable."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", genaiops_prompt),
        ("human", "{input}"),
    ])

    # Create the answer and question chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    print('\nHi, I am NetworkInsights. Ask a question about your network data, or press Enter to exit.\n')

    while True:
        # Prompt user for input
        query = input("Your input: ").strip()

        if query == '':
            print('\nNetworkInsights. Goodbye!\n')
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

        # Use question and answer chain to provide answer
        llm_start = time.perf_counter()
        response = question_answer_chain.invoke({
            "context": matching_docs,
            "input": effective_query
        })
        llm_ms = (time.perf_counter() - llm_start) * 1000

        answer = str(response).strip()
        if not answer:
            print("NetworkInsights: No answer generated for this query.\n")
            total_ms = (time.perf_counter() - query_start) * 1000
            print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms)
            continue

        print('NetworkInsights: ' + answer + '\n')
        total_ms = (time.perf_counter() - query_start) * 1000
        print_timing(retrieval_ms, filter_ms, rerank_ms, llm_ms, total_ms)

    return


if __name__ == "__main__":
    main()
