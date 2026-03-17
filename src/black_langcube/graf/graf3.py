"""
Dummy Graph3 implementation for Black LangCube examples.
This provides a simple mock implementation for strategy generation.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from black_langcube.graf.graph_base import BaseGraph, GraphState

logger = logging.getLogger(__name__)


class Graph3(BaseGraph):
    """Mock Graph3 - Strategy generation."""

    def __init__(
        self, user_message: str, folder_name: str, language: Optional[str] = None
    ):
        super().__init__(GraphState, user_message, folder_name, language)

    @property
    def workflow_name(self):
        return "strategy_generation"

    async def run(self) -> Dict[str, Any]:
        """Mock strategy generation workflow."""
        logger.info("Running Graph3 - Strategy generation")

        await asyncio.sleep(0.1)

        # Create output folder if it doesn't exist
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

        import time

        result = {
            "input_message": self.user_message,
            "language": self.language,
            "workflow": self.workflow_name,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mock_analysis": f"Analyzed '{self.user_message[:30]}...' using {self.workflow_name}",
            "tokens_used": len(self.user_message.split()) * 10,
            "confidence": 0.85,
            "research_strategies": [
                "Literature review approach",
                "Case study analysis",
                "Comparative analysis",
            ],
            "search_terms": ["machine learning", "medical diagnosis", "accuracy"],
            "approach_confidence": 0.82,
            "estimated_research_time": "2-3 hours",
            "processing_nodes": [
                "strategy_planner",
                "approach_selector",
                "term_generator",
            ],
        }

        # Save result to file
        output_file = Path(self.folder_name) / f"{self.workflow_name}_result.json"
        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(result, indent=2))

        logger.info(f"Completed {self.workflow_name}, saved to {output_file}")
        return result
