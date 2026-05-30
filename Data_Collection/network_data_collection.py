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

import json
import logging
import os
import re
import time
from datetime import datetime

import urllib3
import yaml
from catalystcentersdk import api
from dotenv import load_dotenv
from urllib3.exceptions import InsecureRequestWarning

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, "environment.env")
DATA_COLLECTION_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(ENV_PATH)

CC_URL = os.getenv("CC_URL")
CC_USER = os.getenv("CC_USER")
CC_PASS = os.getenv("CC_PASS")
CC_VERSION = os.getenv("CC_VERSION", "2.3.7.9")

APPS_PATH = os.getenv("APPS_PATH")
DATASET = os.getenv("DATASET")

DEVICE_LIST_FILE = os.getenv(
    "DEVICE_LIST_FILE", os.path.join(DATA_COLLECTION_DIR, "device_list.yaml")
)
CLI_COMMANDS_FILE = os.getenv(
    "CLI_COMMANDS_FILE", os.path.join(DATA_COLLECTION_DIR, "cli_commands.yaml")
)

urllib3.disable_warnings(InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)


def normalize_filename_part(value):
    """Normalize text for predictable output file names."""
    value = value.strip().replace(" ", "-")
    value = re.sub(r"[^a-zA-Z0-9_.-]", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value


def ensure_env_config():
    """Validate required environment variables before API calls."""
    required = {
        "CC_URL": CC_URL,
        "CC_USER": CC_USER,
        "CC_PASS": CC_PASS,
        "APPS_PATH": APPS_PATH,
        "DATASET": DATASET,
    }
    for key, value in required.items():
        if not value:
            raise ValueError(f"{key} is not set in environment.env")


def load_device_list(file_path):
    """
    Load device names from YAML.
    Supported formats:
      devices:
        - PDX-RN
        - LO-BN
    or:
      devices:
        - hostname: PDX-RN
        - hostname: LO-BN
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    devices = data.get("devices", [])
    resolved_devices = []
    for item in devices:
        if isinstance(item, str):
            resolved_devices.append(item.strip())
        elif isinstance(item, dict) and item.get("hostname"):
            resolved_devices.append(str(item.get("hostname")).strip())

    resolved_devices = [d for d in resolved_devices if d]
    if not resolved_devices:
        raise ValueError(f"No devices found in {file_path}")
    return resolved_devices


def load_cli_commands(file_path):
    """Load CLI commands from YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    commands = [str(c).strip() for c in data.get("commands", []) if str(c).strip()]
    if not commands:
        raise ValueError(f"No commands found in {file_path}")
    return commands


def find_device_uuid(cc_api, hostname):
    """Find device UUID using hostname."""
    response = cc_api.devices.get_device_list(hostname=hostname)
    devices = response.get("response", []) if isinstance(response, dict) else []
    if not devices:
        return None, None
    device = devices[0]
    return device.get("id"), device.get("hostname") or hostname


def wait_for_task_completion(cc_api, task_id, timeout_seconds=180):
    """Poll a Catalyst Center task until completion."""
    end_time = time.time() + timeout_seconds
    while time.time() < end_time:
        task = cc_api.task.get_task_by_id(task_id=task_id).get("response", {})
        if task.get("isError"):
            return task, False
        if task.get("endTime"):
            return task, True
        time.sleep(2)
    return {}, False


def extract_command_output(file_content_data, command):
    """Parse command output from Catalyst Center file API payload."""
    try:
        payload = json.loads(file_content_data)
        if not payload:
            return ""
        success = payload[0].get("commandResponses", {}).get("SUCCESS", {})
        if command in success:
            return success[command]
        if success:
            # If the command key differs slightly, return first successful output.
            return next(iter(success.values()))
    except Exception:
        return file_content_data
    return ""


def collect_device_data(cc_api, dataset_path, hostname, commands):
    """Run command list for one device and write outputs to DATASET."""
    device_id, resolved_name = find_device_uuid(cc_api, hostname)
    if not device_id:
        logging.warning("Device not found in Catalyst Center inventory: %s", hostname)
        return

    logging.info("Collecting data for device: %s", resolved_name)
    for command in commands:
        logging.info("  Running command: %s", command)
        run_response = cc_api.command_runner.run_read_only_commands_on_devices(deviceUuids=[device_id], commands=[command])
        task_id = run_response.get("response", {}).get("taskId")
        if not task_id:
            logging.warning("  Could not start command task for: %s", command)
            continue

        task_status, completed = wait_for_task_completion(cc_api, task_id=task_id)
        if not completed:
            logging.warning("  Command task timed out/failed for: %s", command)
            continue

        progress = task_status.get("progress", "")
        try:
            progress_json = json.loads(progress) if progress else {}
            file_id = progress_json.get("fileId")
        except Exception:
            file_id = None

        if not file_id:
            logging.warning("  No output file id for command: %s", command)
            continue

        # Prefer the non-deprecated SDK method; keep fallback for older SDK builds.
        if hasattr(cc_api.file, "download_a_file_by_file_id"):
            file_content = cc_api.file.download_a_file_by_file_id(file_id=file_id).data
        else:
            file_content = cc_api.file.download_a_file_by_fileid(file_id=file_id).data
        file_content_data = file_content.decode("utf-8", errors="ignore")
        command_output = extract_command_output(file_content_data, command)

        command_for_file = normalize_filename_part(command)
        filename = f"{resolved_name}_{command_for_file}.txt"
        filepath = os.path.join(dataset_path, filename)

        output_data = (
            f"Device hostname - {resolved_name}\n"
            f"CLI Command - {command}\n"
            f"{command_output}\n"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output_data)

        logging.info("  Saved: %s", filepath)


def main():
    """
    Collect device CLI outputs from Catalyst Center using:
    - device list from Data_Collection/device_list.yaml
    - command list from Data_Collection/cli_commands.yaml
    Save output files in DATASET for embeddings ingestion.
    """
    ensure_env_config()

    dataset_path = os.path.join(APPS_PATH, DATASET)
    os.makedirs(dataset_path, exist_ok=True)

    if not os.path.isfile(DEVICE_LIST_FILE):
        raise ValueError(f"Device list file not found: {DEVICE_LIST_FILE}")
    if not os.path.isfile(CLI_COMMANDS_FILE):
        raise ValueError(f"CLI commands file not found: {CLI_COMMANDS_FILE}")

    devices = load_device_list(DEVICE_LIST_FILE)
    commands = load_cli_commands(CLI_COMMANDS_FILE)

    logging.info("Data collection start: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Catalyst Center URL: %s", CC_URL)
    logging.info("Dataset folder: %s", dataset_path)
    logging.info("Devices to query: %s", ", ".join(devices))
    logging.info("Commands to run: %s", ", ".join(commands))

    cc_api = api.CatalystCenterAPI(
        username=CC_USER,
        password=CC_PASS,
        base_url=CC_URL,
        version=CC_VERSION,
        verify=False,
    )

    for hostname in devices:
        collect_device_data(cc_api, dataset_path=dataset_path, hostname=hostname, commands=commands)

    logging.info("Data collection end: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
