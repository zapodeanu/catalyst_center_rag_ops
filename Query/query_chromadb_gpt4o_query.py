#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Cisco and/or its affiliates.
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

__author__ = "Gabriel Zapodeanu TME, ENB"
__email__ = "gzapodea@cisco.com"
__version__ = "0.1.0"
__copyright__ = "Copyright (c) 2025 Cisco and/or its affiliates."
__license__ = "Cisco Sample Code License, Version 1.1"

import os


import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv('environment.env')

# database server details
DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = int(os.getenv('DB_PORT'))
DB_COLLECTION = os.getenv('DB_COLLECTION')

# OpenAI key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

OPENAI_MODEL = 'gpt-4o'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = os.getenv('MODEL_NAME')

def main():
    """
    A query app for network troubleshooting, powered by LangChain, that retrieves similarity matches
    from Chroma and generates responses using OpenAI's GPT-4o.
    """

    # Chroma DB server details and connection
    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # Define the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # Chroma DB connection to server and collection
    chroma_db = Chroma(
        client=chroma_db_server,
        collection_name=DB_COLLECTION,
        embedding_function=embeddings
    )

    # Define retriever from Chroma DB and number of proximity matches
    retriever = chroma_db.as_retriever(search_kwargs={"k":8})

    # Define the LLM used - OpenAI, model 'gpt-4o'
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=1)

    # Create the prompt template
    genaiops_prompt = (
        "You are an assistant for network insights tasks."
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

    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print('\nHi, I am your InsightsPilot! Enter your query or press Enter to end.\n')

    while True:
        # Prompt user for input
        query = input("Your input: ").strip()

        if query == '':
            print('\nInsightsPilot. Goodbye!\n')
            break

        # Generate response using RAG chain
        response = rag_chain.invoke({"input": query})

        # Debug: show the proximity matches
        '''
        print("\n--- DEBUG: Retrieved Documents ---")
        for i, doc in enumerate(response['context']):
            print(f"Doc {i + 1}: {doc.page_content[:200]}...")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"Metadata: {doc.metadata}")
        print("--- End Retrieved Documents ---\n")
        '''

        print('InsightsPilot: ' + response['answer'] + '\n')

    return


if __name__ == "__main__":
    main()
