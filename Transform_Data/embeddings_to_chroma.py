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

# logging, info level
logging.basicConfig(level=logging.INFO)
# create loggers as warning for chromadb, sentence transformers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('chromadb.telemetry').setLevel(logging.WARNING)

load_dotenv('environment.env')

# database server details
DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = int(os.getenv('DB_PORT'))
DB_COLLECTION = os.getenv('DB_COLLECTION')
APPS_PATH = os.getenv('APPS_PATH')
DATASET = os.getenv('DATASET')
MODEL_NAME = os.getenv('MODEL_NAME')


def load_docs(directory):
    """
    This function will load the docs from the specified folder
    :param directory: the data to be embedded
    :return: documents from folder
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(document, chunk_size, chunk_overlap, separator, file):
    """
    This function will split the documents with the defined number of characters, overlap,
    and separator. It will add metadata to each chunk. The metadata will be created based
    on the filename.
    :param document: document to be split
    :param chunk_size: chuck size
    :param chunk_overlap: overlap
    :param separator: separator
    :param file: filename for the content
    :return: doc split in chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   separators=separator)
    split_documents = text_splitter.split_documents(document)

    # collect the data for device, issue, command, to be used in metadata
    file_details = file.split('_')

    device_name = file_details[0]
    command = file_details[1].replace('-', ' ')

    chunk_number = 1
    for doc in split_documents:
        doc.metadata['chunk_number'] = chunk_number  # Add a chunk number as metadata
        doc.metadata['device name'] = device_name
        doc.metadata['CLI command'] = command
        chunk_number += 1

    return split_documents


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


# noinspection PyProtectedMember,PyUnusedLocal
def create_doc_embeddings(document, file):
    """
    The function will create the embeddings for the {doc}, with the metadata provided, using
    {sentence-transformers/MODEL_NAME} model.
    Update the ChromaDB vector database with the new embeddings
    :param document: document to be embedded
    :param file: filename for the document
    :return: collection count, after updating it
    """

    # connection to the Chroma DB server
    chroma_db_server = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # split the document, create embeddings
    docs = split_docs(document=document, chunk_size=100, chunk_overlap=25, separator='!', file=file)

    # define embeddings model
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
    This application will load the files from the {DATASET} folder.
    Each file will be split in chunks, metadata will be created for each chunk, and
    embeddings will be created for each chunk.
    The embeddings will be uploaded to the Chroma DB server.
    """

    logging.info(' The folder with the data to be embedded is: ' + DATASET)

    os.chdir(APPS_PATH)
    documents = load_docs(DATASET)
    logging.info(' There are ' + str(len(documents)) + ' documents in the folder')

    # create the chroma client, and create or get the collection
    chroma_db = chromadb.HttpClient(host=DB_SERVER, port=DB_PORT)

    # chromadb heartbeat
    chroma_db.heartbeat()

    # load the files from the folder
    files_list = os.listdir(DATASET)
    logging.info(' We will create vector representations for these files: ')

    # for each file create and update the embeddings
    for file in files_list:
        logging.warning('    ' + file)
        file_content = load_file(file, DATASET)
        filename = file.split(".")[0]
        collection_count = create_doc_embeddings(document=file_content, file=filename)
        logging.info(' Collection count is ' + str(collection_count))

    # chromadb heartbeat
    chroma_db.heartbeat()


if __name__ == "__main__":
    main()
