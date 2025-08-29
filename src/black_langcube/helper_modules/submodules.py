"""
This module provides helper functions for session management and process termination messaging.
Functions:
    SessionCreator():
        Creates a timestamped session folder inside the "results" directory.
        Returns a dictionary with the absolute path of the created folder.
        Logs relevant information and raises a RuntimeError if folder creation fails.
    end_process(user_message, folder_name, language):
        Ends the process and generates a message in the specified language.
        Returns a message indicating the process has ended, translated as needed.
        Logs errors if required arguments are missing.
"""

import logging
logger = logging.getLogger(__name__)

from datetime import datetime
from pathlib import Path
from black_langcube.messages.message_end_process import message_end_process

def SessionCreator():
    """
    Creates a session folder with a timestamped name inside the "results" directory.
    This function generates a folder name based on the current timestamp, creates the folder
    (including any necessary parent directories), and returns the absolute path of the folder
    as part of a dictionary. If the folder creation fails, it logs a critical error and raises
    a RuntimeError.
    Returns:
        dict: A dictionary containing the absolute path of the created folder under the key "folder_name".
    Raises:
        RuntimeError: If the folder creation fails for any reason.
    """

    logger.info("----- Session Node -----")

    # Get the current time as a formatted string
    formatted_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Formatted Timestamp: %s", formatted_timestamp)

    # Define the folder name
    folder_name = Path("./results") / formatted_timestamp

    # Create the folder
    try:
        folder_name.mkdir(parents=True, exist_ok=True)  # parents=True creates parent dirs if needed
        logger.info(f"Folder '{folder_name}' created successfully!")
        absolute_path = folder_name.resolve()
        
        # Add folder_name to the state
        folder = str(absolute_path)
        return {"folder_name": str(folder)}  # Return folder_name
    except Exception as e:
        #print(f"An error occurred: {e}")
        #return {"messages": [""], "folder_name": None}
        logger.critical("Failed to create session folder")
        raise RuntimeError("Failed to create session folder") from e


def end_process(user_message, folder_name, language):
    """
    End the process and generate a message based on the provided language.
    Args:
        user_message (str): The user's query or message to include in the response.
        folder_name (str): The name of the folder associated with the process.
        language (str): The language in which the response message should be generated.
    Returns:
        str: A message indicating the process has ended, translated into the specified language.
    Notes:
        - If `folder_name` is not provided, the function logs an error and returns a default message.
        - If `language` is not provided, the function logs an error and returns a default message.
        - For "Czech" and "English" languages, predefined messages are used.
        - For other languages, a translation service (`TUsubgraph.invoke`) is used to generate the message.
    """

    if not folder_name:
        logger.error("No folder name provided for end_process function")
        return "Returning from end_process function due to missing folder name."

    if not language:
        logger.error("No language specified for end_process function")
        return "Returning from end_process function due to missing language."

    message = message_end_process(language, folder_name, output_filename=None)
    logger.info("Ending the process for folder: %s", folder_name)

    return message
