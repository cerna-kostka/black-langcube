"""
Calculates and aggregates token usage and costs from multiple graph output subfolders within a specified folder.
This function processes the outputs of five graph runs (graph1 to graph5) located in subfolders of the given `folder_name`.
It sums up the input tokens, output tokens, and token costs by extracting values from each graph's output JSON file using
the `get_result_from_graph_outputs` helper function. The function also records the process start and end times, calculates
the duration, and writes a summary of the results to a `tokens.txt` file in the specified folder.
Args:
    folder_name (str or Path): The path to the folder containing the graph output subfolders (graph1 to graph5).
Returns:
    tuple: A tuple containing:
        - tokens_in (int): Total number of input tokens across all graphs.
        - tokens_out (int): Total number of output tokens across all graphs.
        - tokens_cost (float): Total cost of tokens across all graphs.
        - process_duration (float): Duration of the process in seconds.
Side Effects:
    - Writes a summary file named `tokens.txt` in the specified folder, containing token counts, costs, and timing information.
    - Logs information about the process using the module logger.
Raises:
    Any exceptions raised by file I/O, JSON parsing, or the helper functions are propagated.
"""

import logging
logger = logging.getLogger(__name__)

import os
from pathlib import Path
from datetime import datetime

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs
from black_langcube.helper_modules.calculate_duration import calculate_duration_seconds

def TokenCostsCount(folder_name):
    """
    This function calculates the token costs for a given folder containing graph outputs.
    It aggregates the token counts from multiple subfolders and writes the results to a file.
    """
    
    logger.info(f"Calculating token costs for folder: {folder_name}")
    
    subfolder1_name = Path(folder_name) / "graph1"
    subfolder2_name = Path(folder_name) / "graph2"
    subfolder3_name = Path(folder_name) / "graph3"
    subfolder4_name = Path(folder_name) / "graph4"
    subfolder5_name = Path(folder_name) / "graph5"

    tokens_in = (
        get_result_from_graph_outputs("token_counter", "", "graph1_tokens_in", "", subfolder1_name, "graph1_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph2_tokens_in", "", subfolder2_name, "graph2_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph3_tokens_in", "", subfolder3_name, "graph3_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph4_tokens_in", "", subfolder4_name, "graph4_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph5_tokens_in", "", subfolder5_name, "graph5_output.json")
    )

    tokens_out = (
        get_result_from_graph_outputs("token_counter", "", "graph1_tokens_out", "", subfolder1_name, "graph1_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph2_tokens_out", "", subfolder2_name, "graph2_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph3_tokens_out", "", subfolder3_name, "graph3_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph4_tokens_out", "", subfolder4_name, "graph4_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph5_tokens_out", "", subfolder5_name, "graph5_output.json")
    )

    tokens_cost = (
        get_result_from_graph_outputs("token_counter", "", "graph1_tokens_price", "", subfolder1_name, "graph1_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph2_tokens_price", "", subfolder2_name, "graph2_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph3_tokens_price", "", subfolder3_name, "graph3_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph4_tokens_price", "", subfolder4_name, "graph4_output.json") +
        get_result_from_graph_outputs("token_counter", "", "graph5_tokens_price", "", subfolder5_name, "graph5_output.json")
    )

    time_beginning = Path(folder_name).name 
    time_ending = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    process_duration = calculate_duration_seconds(time_beginning, time_ending)

    token_file_path = os.path.join(folder_name, "tokens.txt")
    token_file = open(token_file_path, "w", encoding="utf-8")
    token_file.write("Tokens In: " + str(tokens_in) + "\n")
    token_file.write("Tokens Out: " + str(tokens_out) + "\n")
    token_file.write("Tokens Cost: " + str(tokens_cost) + "\n")
    token_file.write("Process Start Time: " + str(time_beginning) + "\n")
    token_file.write("Process End Time: " + str(time_ending) + "\n")
    token_file.write("Process Duration: " + str(process_duration) + " seconds\n")
    token_file.close()
    logger.info(f"Token amounts, costs and duration data saved to {token_file_path}")
    return tokens_in, tokens_out, tokens_cost, process_duration

# Example usage - or for debugging  
"""if __name__ == "__main__":
    folder_name = "./results/2025-04-03 14:15:06"
    tokens_in, tokens_out, tokens_cost = TokenCostsCount(folder_name)
    print(f"Tokens In: {tokens_in}")
    print(f"Tokens Out: {tokens_out}")
    print(f"Tokens Cost: {tokens_cost}")"""