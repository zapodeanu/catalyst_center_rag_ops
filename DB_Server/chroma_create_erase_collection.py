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

import logging
import chromadb
import time
import os

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, 'environment.env')
load_dotenv(ENV_PATH)

os.environ['TZ'] = 'America/Los_Angeles'  # define the timezone for PST
time.tzset()  # adjust the timezone, more info https://help.pythonanywhere.com/pages/SettingTheTimezone/

# logging, debug level, to file {application_run.log}
logging.basicConfig(level=logging.INFO)

DB_SERVER = os.getenv('DB_SERVER')
DB_PORT = os.getenv('DB_PORT')
DB_COLLECTION = os.getenv('DB_COLLECTION')


# noinspection PyUnusedLocal
def main():
    """
    This app will ask the user for input for creating or deleting a collection from
    the Chroma DB server
    User input: Delete (d) or Create (c) collection
    """

    # configure the Chroma DB server
    chroma_client = chromadb.HttpClient(host=DB_SERVER, port=int(DB_PORT))

    # create or erase the collection
    print(f"Target Chroma server: {DB_SERVER}:{DB_PORT}")
    print(f"Target collection: {DB_COLLECTION}")
    user_input = input('Delete (d) or Create (c) collection? ').strip().lower()
    if user_input == 'd':
        confirm = input(
            f"Type YES to confirm deleting collection '{DB_COLLECTION}': "
        ).strip()
        if confirm != "YES":
            logging.info("Delete cancelled by user.")
            return
        try:
            chroma_client.delete_collection(name=DB_COLLECTION)
            logging.info("Collection deleted successfully.")
        except Exception as err:
            logging.error("Unable to delete collection '%s': %s", DB_COLLECTION, err)
    elif user_input == 'c':
        try:
            chroma_client.get_or_create_collection(name=DB_COLLECTION)
            logging.info("Collection is ready (already existed or was created).")
        except Exception as err:
            logging.error("Unable to create/get collection '%s': %s", DB_COLLECTION, err)
    else:
        logging.error("Invalid option '%s'. Use 'd' to delete or 'c' to create.", user_input)
        return

    # chromadb heartbeat
    chroma_client.heartbeat()


if __name__ == "__main__":
    main()
