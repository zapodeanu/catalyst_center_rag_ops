# NetworkInsights RAG for Cisco Catalyst Center

NetworkInsights is a Retrieval-Augmented Generation (RAG) toolkit for network operations.
It collects CLI outputs from Cisco Catalyst Center, builds embeddings in Chroma, and provides
interactive query/conversation clients for OpenAI and Anthropic models.
Some code updated with Cursor Codex 5.3.

## Repository Structure

- `DB_Server/`
  - `chroma_server.py` - starts the Chroma server.
  - `chroma_create_erase_collection.py` - create/delete collection utility.
- `Data_Collection/`
  - `network_data_collection.py` - collects CLI outputs from listed devices.
  - `device_list.yaml` - devices to query.
  - `cli_commands.yaml` - commands to execute.
- `Transform_Data/`
  - `embeddings_to_chroma.py` - splits documents, adds metadata, uploads embeddings.
- `Client_Apps/`
  - Classic clients:
    - `query_network_insights_openai.py`
    - `conversation_network_insights_openai.py`
    - `conversation_network_insights_anthropic.py`
  - LCEL clients:
    - `query_network_insights_openai_lcel.py`
    - `conversation_network_insights_openai_lcel.py`
    - `conversation_network_insights_anthropic_lcel.py`
  - Test helper:
    - `test_lcel_versions.py`
- `DATASET/` - raw text data used for embeddings.
- `VDB/` - local Chroma vector database storage.
- `environment.env` - local runtime configuration (do not commit secrets).

## Features

- End-to-end network data flow:
  1. Collect device CLI data from Catalyst Center
  2. Build metadata-rich embeddings
  3. Query with RAG clients
- Metadata-aware retrieval (minimal filter):
  - device filter based on explicit hostname in query
- Model/provider flexibility:
  - OpenAI and Anthropic clients
  - Classic and LCEL variants
- Enterprise-friendly HF TLS handling:
  - supports custom CA bundle via env configuration

## Requirements

- Python 3.10+
- Cisco Catalyst Center access (for data collection)
- OpenAI and/or Anthropic API credentials (for client apps)

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `environment.env` (local only):
- API keys
- Catalyst Center credentials
- Chroma settings
- model settings

Recommended keys:

```env
OPENAI_MODEL='gpt-5.5'
CLAUDE_MODEL='claude-sonnet-4-6'
MODEL_NAME='all-MiniLM-L6-v2'
```

Optional keys:

```env
MODEL_LOCAL_PATH='/absolute/path/to/local/embedding/model'
HF_CA_BUNDLE='/absolute/path/to/ca-bundle.pem'
```

## Typical Workflow

### 1) Start Chroma

```bash
.venv/bin/python DB_Server/chroma_server.py
```

### 2) Create collection (if needed)

```bash
.venv/bin/python DB_Server/chroma_create_erase_collection.py
```

### 3) Collect network data

Update:
- `Data_Collection/device_list.yaml`
- `Data_Collection/cli_commands.yaml`

Run:

```bash
.venv/bin/python Data_Collection/network_data_collection.py
```

### 4) Build embeddings

```bash
.venv/bin/python Transform_Data/embeddings_to_chroma.py
```

### 5) Run clients

Classic query:

```bash
.venv/bin/python Client_Apps/query_network_insights_openai.py
```

Classic conversations:

```bash
.venv/bin/python Client_Apps/conversation_network_insights_openai.py
.venv/bin/python Client_Apps/conversation_network_insights_anthropic.py
```

LCEL clients:

```bash
.venv/bin/python Client_Apps/query_network_insights_openai_lcel.py
.venv/bin/python Client_Apps/conversation_network_insights_openai_lcel.py
.venv/bin/python Client_Apps/conversation_network_insights_anthropic_lcel.py
```

## LCEL Smoke Test

Run all LCEL clients with predefined prompts:

```bash
.venv/bin/python Client_Apps/test_lcel_versions.py
```

## Notes

- `environment.env` is for local development and should not contain shared secrets.
- HF warning about unauthenticated requests is non-blocking for most POC usage.
- For best retrieval precision, include device hostname explicitly in queries.
