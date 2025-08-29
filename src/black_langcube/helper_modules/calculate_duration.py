"""
This module provides a utility function to calculate the duration in seconds between two datetime strings.
Functions:
    calculate_duration_seconds(start_time_str, end_time_str):
        Calculates the duration in seconds between two datetime strings formatted as 'YYYY-MM-DD HH:MM:SS'.
        Useful for determining the elapsed time of a process, such as from a user's first submission to the output of files.
"""
from datetime import datetime
        
def calculate_duration_seconds(start_time_str, end_time_str):
    """
    Calculate the duration in seconds between two datetime strings.
    Used for calculating the duration of an overall process of making the review from the user's first submission to outputting the files.
    
    Args:
        start_time_str (str): Start time in format 'YYYY-MM-DD HH:MM:SS'
        end_time_str (str): End time in format 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
        float: Duration in seconds
    """
    
    # Parse the time strings into datetime objects
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    
    # Calculate the difference
    duration = end_time - start_time
    
    # Return the total duration in seconds
    return duration.total_seconds()

# Example
"""start_time = "2025-04-12 17:23:35"
end_time = "2025-04-12 17:54:15"

duration_seconds = calculate_duration_seconds(start_time, end_time)
print(f"Duration: {duration_seconds} seconds")  # Output: Duration: 1840.0 seconds
print(f"Duration: {duration_seconds/60:.2f} minutes")  # Output: Duration: 30.67 minutes"""