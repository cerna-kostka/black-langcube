def get_basegraph_classes():
    """
    Imports and returns the BaseGraph and GraphState classes from the graf.graph_base module.
    Serves as a utility to avoid circular imports in the project.

    Returns:
        tuple: A tuple containing the BaseGraph and GraphState classes.
    """
    from black_langcube.graf.graph_base import BaseGraph, GraphState
    return BaseGraph, GraphState