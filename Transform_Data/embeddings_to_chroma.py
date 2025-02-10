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

import logging
import os
import time

import chromadb
from dotenv import load_dotenv
# noinspection PyProtectedMember
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['TZ'] = 'America/Los_Angeles'  # define the timezone for PST
time.tzset()  # adjust the timezone, more info https://help.pythonanywhere.com/pages/SettingTheTimezone/

# logging, Warning level, to file
logging.basicConfig(level=logging.WARNING)

load_dotenv('environment.env')

MODEL_NAME = 'all-MiniLM-L6-v2'

# database server details
DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = int(os.getenv('DB_PORT'))
DB_COLLECTION = os.getenv('DB_COLLECTION')
DB_PATH = os.getenv('DB_PATH')

OPS_DATA = 'OPS_DATA'


def load_docs(directory):
    """
    This function will load the docs from the specified folder
    :param directory: folder with the data to be embedded
    :return: documents from folder
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(document, chunk_size, chunk_overlap, separator):
    """
    This function will split the documents with the defined number of characters, overlap, and separator
    :param document: document to be split
    :param chunk_size: chuck size
    :param chunk_overlap: overlap
    :param separator: separator
    :return: doc split in chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separator)
    docs = text_splitter.split_documents(document)
    return docs


def load_file(filename, path):
    """
    The function will load the file in the folder
    :param filename: name of file
    :param path: folder path
    :return: file content
    """
    loader = TextLoader(path + '/' + filename)
    file_content = loader.load()
    return file_content


def create_doc_embeddings(document):
    """
    The function will create the embeddings for the {doc}, with the metadata provided, using
    {sentence-transformers/all-MiniLM-L6-v2} model.
    Update the ChromaDB vector database with the new embeddings
    :param document: document to be embedded
    :return: collection count, after updating it
    """

    # connection to the Chroma DB server
    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # split the document, create embeddings
    docs = split_docs(document=document, chunk_size=100, chunk_overlap=25, separator="!")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # update the chroma db collection with the new embeddings
    chroma_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=chroma_db_server,
        collection_name=DB_COLLECTION
        )

    # get the updated collection count
    chroma_collection = Chroma(
        client=chroma_db_server,
        collection_name=DB_COLLECTION
    )
    return chroma_collection._collection.count()


def main():
    """
    This application will load the files from the {OPS_DATA} folder.
    Each file will be split in chunks, and embeddings will be created for each chunk.
    The embeddings and chunks will be uploaded to the Chroma DB server.
    """

    logging.warning(' The folder with the data to be embedded is: ' + OPS_DATA)

    documents = load_docs(OPS_DATA)
    logging.warning(' There are ' + str(len(documents)) + ' documents in the folder')

    # create the chroma client, and create or get the collection
    chroma_db = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # chromadb heartbeat
    chroma_db.heartbeat()

    # load the files from the folder
    files_list = os.listdir(OPS_DATA)
    logging.warning(' The files to be embedded are: ')

    # for each file create and update the embeddings
    for file in files_list:
        logging.warning('    ' + file)
        file_content = load_file(file, OPS_DATA)
        filename = [file.split(".")[0]]
        collection_count = create_doc_embeddings(document=file_content)
        logging.warning(' Collection count is ' + str(collection_count))

    # chromadb heartbeat
    chroma_db.heartbeat()


if __name__ == "__main__":
    main()
