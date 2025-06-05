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

from dotenv import load_dotenv


load_dotenv('environment.env')

os.environ['TZ'] = 'America/Los_Angeles'  # define the timezone for PST
time.tzset()  # adjust the timezone, more info https://help.pythonanywhere.com/pages/SettingTheTimezone/

# logging, debug level, to file {application_run.log}
logging.basicConfig(level=logging.INFO)

# database server details
DB_PATH = os.getenv('DB_PATH')
DB_PORT = os.getenv('DB_PORT')


def main():
    """
    This application will start a new Chroma DB server.
    It requires the server port and the path where to save tge database
    """
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    os.system('chroma run --port ' + DB_PORT + ' --path ' + DB_PATH)


if __name__ == "__main__":
    main()



