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

__author__ = "Gabriel Zapodeanu PTME"
__email__ = "gzapodea@cisco.com"
__version__ = "0.1.0"
__copyright__ = "Copyright (c) 2025 Cisco and/or its affiliates."
__license__ = "Cisco Sample Code License, Version 1.1"

import os
import warnings

import chromadb
from dotenv import load_dotenv
# noinspection PyProtectedMember
from langchain._api import LangChainDeprecationWarning
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)  # to disable LangChain warnings

os.chdir('../')

load_dotenv('environment.env')

# database server details
DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = int(os.getenv('DB_PORT'))
DB_COLLECTION = os.getenv('DB_COLLECTION')

# OpenAi key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

MODEL_NAME = 'all-MiniLM-L6-v2'
OPENAI_MODEL = 'gpt-3.5-turbo'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    The app will run a LangChain that will identify similarity results with a query.
    The match will be provided to the user after running through the GPT-3.5 OpenAI model.
    """

    # Chroma DB server details
    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # define the model for creating embeddings for the query
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # initialize the Chroma DB connection to server and collection
    chroma_db = Chroma(
        client=chroma_db_server,
        collection_name=DB_COLLECTION,
        embedding_function=embeddings
    )

    # define the LLM used - OpenAI, model 'gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=1)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)

    query = "y"
    while query != ' ':
        # prompt the user for the query input
        query = input('\n\n\n\nI am a NetOps bot. How may I help you?  ')

        # search for the similarity matches from the Chroma DB server/collection
        matching_docs = chroma_db.similarity_search(query)

        # The similarity search result and query will be used to generate the answer using the LLM
        answer = chain.run(input_documents=matching_docs, question=query, set_verbose=False)
        print(answer)
    return


if __name__ == "__main__":
    main()
