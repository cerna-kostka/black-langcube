"""
This module provides utility functions to extract specific results from JSON-formatted files.
It includes functions to retrieve nested or simple values from JSON objects stored line-by-line
in files, with robust error handling for missing files, invalid JSON, or missing keys.
Functions:
    get_result_from_graph_outputs(key1, key2, subkey, subsubkey, subfolder_name, filename):
        Retrieves a specific nested value from a JSON-formatted file based on provided keys.
        Handles errors such as missing files, invalid JSON, or missing keys, and returns
        descriptive error messages as dictionaries.
    get_simple_result_from_graph_outputs(key, subfolder_name, filename):
        Retrieves a value associated with a given key from JSON objects in a file.
        Handles errors such as missing files, invalid JSON, or missing keys, and returns
        descriptive error messages as dictionaries.

"""

import logging
logger = logging.getLogger(__name__)
import os
import json


def get_result_from_graph_outputs(key1, key2, subkey, subsubkey, subfolder_name, filename):
    """
    Retrieves a specific result from a JSON-formatted file based on the provided keys.
    This function reads a file line by line, parses JSON objects, and searches for 
    specific nested keys to extract a value. If the file does not exist, contains 
    invalid JSON, or the specified keys are not found, appropriate error messages 
    are returned.
    Args:
        key1 (str): The primary key to search for in the JSON objects.
        key2 (str): The secondary key to search for in the JSON objects.
        subkey (str): The subkey to search for within the data of `key1` or `key2`.
        subsubkey (str): The sub-subkey to search for within the data of `key2[subkey]`.
        subfolder_name (str): The name of the subfolder containing the file.
        filename (str): The name of the file to read and parse.
    Returns:
        dict or any:
            - If successful, returns the value associated with the specified keys.
            - If an error occurs (e.g., file not found, invalid JSON, or no matching entry), 
              returns a dictionary with an "error" key and a descriptive error message.
    Errors:
        - {"error": "subfolder_name is not set."}: If `subfolder_name` is not provided.
        - {"error": "<file_path> not found."}: If the specified file does not exist.
        - {"error": "Invalid JSON in line: <line>."}: If a line in the file contains invalid JSON.
        - {"error": "No entry found."}: If no matching entry is found in the file.
    Example:
        result = get_result_from_graph_outputs(
            key1="data1",
            key2="data2",
            subkey="subdata",
            subsubkey="subsubdata",
            subfolder_name="results",
            filename="output.json"
        )
    """

    if not subfolder_name:
        logger.error("subfolder_name is not set.")
        return {"error": "subfolder_name is not set."}
    
    file_path = os.path.join(subfolder_name, filename)
    answer = None  # This will hold the "result" value once we find it

    if os.path.exists(file_path):
        # 1) Read the file line by line and parse possible JSON objects
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if key1 in data:
                            key1_data = data.get(key1)
                            if key1_data and subkey in key1_data:
                                answer = key1_data.get(subkey)
                        if key2 in data:
                            key2_data = data.get(key2)
                            if key2_data and subkey in key2_data:
                                subkey_data = key2_data.get(subkey)
                                if subkey_data and subsubkey in subkey_data:
                                    answer = subkey_data.get(subsubkey)
                    except json.JSONDecodeError:
                        # Ignore lines that are not valid JSON
                        #pass
                        logger.error(f"Invalid JSON in line: {line}")
                        return {"error": f"Invalid JSON in line: {line}."}
    else:
        logger.error(f"{file_path} not found.")
        return {"error": f"{file_path} not found."}

    # If no matching entry was found, answer will remain None
    if answer is None:
        return {"error": f"No entry found."}

    return answer

def get_simple_result_from_graph_outputs(key, subfolder_name, filename):
    """
    Retrieve a value associated with a given key from JSON objects in a file.
    This function searches for a specified key in JSON objects found line-by-line in a file
    located at the path constructed from `subfolder_name` and `filename`. If the key is found,
    its value is returned. If the file does not exist, the subfolder name is not set, or no
    matching entry is found, an error dictionary is returned.
    Args:
        key (str): The key to search for in each JSON object.
        subfolder_name (str): The name of the subfolder containing the file.
        filename (str): The name of the file to read.
    Returns:
        Any: The value associated with the specified key if found.
        dict: An error dictionary if the subfolder name is not set, the file is not found,
              the JSON is invalid, or the key is not found in any JSON object.
    """

    if not subfolder_name:
        logger.error("subfolder_name is not set.")
        return {"error": "subfolder_name is not set."}
    
    file_path = os.path.join(subfolder_name, filename)
    answer = None  # This will hold the "result" value once we find it

    if os.path.exists(file_path):
        # 1) Read the file line by line and parse possible JSON objects
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if key in data:
                            answer = data[key]
                    except json.JSONDecodeError:
                        # Ignore lines that are not valid JSON
                        logger.error(f"Invalid JSON in line: {line}")
                        return {"error": f"Invalid JSON in line: {line}."}
    else:
        logger.error(f"{file_path} not found.")
        return {"error": f"{file_path} not found."}

    # If no matching entry was found, answer will remain None
    if answer is None:
        return {"error": f"No entry found."}

    return answer