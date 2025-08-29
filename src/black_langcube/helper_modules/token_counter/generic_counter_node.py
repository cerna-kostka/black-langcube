"""
This module provides a factory function to create a token counter node for use in a workflow.
Functions:
    create_token_counter_node(token_keys, workflow_prefix):
        Factory function that returns a node function for counting tokens.
        The returned node function takes (state, config) as arguments and returns a dictionary
        with token counts, where the keys are prefixed by the provided workflow_prefix.
        Args:
            token_keys (list): List of keys specifying which tokens to count.
            workflow_prefix (str): Prefix to use for the output dictionary keys.
        Returns:
            function: A node function that computes token counts from the given state.
                "<workflow_prefix>_tokens_in": ...,
                "<workflow_prefix>_tokens_out": ...,
                "<workflow_prefix>_tokens_price": ...
"""

import logging
logger = logging.getLogger(__name__)

def create_token_counter_node(token_keys, workflow_prefix):
    """
    Returns a node function that uses the provided token_keys and workflow_prefix.
    
    The returned function takes (state, config) and returns a dict with token counts keyed by workflow_prefix.
    
    Example output:
      {
          "graph1_tokens_in": ...,
          "graph1_tokens_out": ...,
          "graph1_tokens_price": ...
      } 
    """

    logger.info(f"Creating token counter node with prefix: {workflow_prefix} and keys: {token_keys}")

    def token_counter_node(state, config):
        from black_langcube.helper_modules.token_counter.token_counter import TokenCounter  # import TokenCounter here if needed
        counter = TokenCounter(token_keys)
        result = counter.count_tokens(state)
        return {
            f"{workflow_prefix}_tokens_in": result["tokens_in"],
            f"{workflow_prefix}_tokens_out": result["tokens_out"],
            f"{workflow_prefix}_tokens_price": result["tokens_price"],
        }
    return token_counter_node