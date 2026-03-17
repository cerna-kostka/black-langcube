"""
Dummy Graph1 implementation for Black LangCube examples.
This replaces the complex graph1.py with a simple mock implementation.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from black_langcube.graf.graph_base import BaseGraph

logger = logging.getLogger(__name__)


class Graph1(BaseGraph):
    """Mock Graph1 - Question processing and language detection."""

    def __init__(
        self, user_message: str, folder_name: str, language: Optional[str] = None
    ):
        # Import GraphState here to avoid circular imports
        from black_langcube.graf.graph_base import GraphState

        super().__init__(GraphState, user_message, folder_name, language)

    @property
    def workflow_name(self):
        return "question_processing"

    async def run(self) -> Dict[str, Any]:
        """Mock question processing workflow."""
        logger.info("Running Graph1 - Question processing and language detection")

        await asyncio.sleep(0.1)

        # Create output folder if it doesn't exist
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

        # Generate mock result
        import time

        result = {
            "input_message": self.user_message,
            "language": self.language,
            "workflow": self.workflow_name,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mock_analysis": f"Analyzed '{self.user_message[:30]}...' using {self.workflow_name}",
            "tokens_used": len(self.user_message.split()) * 10,
            "confidence": 0.85,
            "detected_language": self.language,
            "question_type": "informational"
            if "what" in self.user_message.lower()
            else "analytical",
            "entities": ["AI", "healthcare", "technology"],
            "complexity_score": 0.7,
            "processing_nodes": [
                "language_detector",
                "entity_extractor",
                "question_classifier",
            ],
        }

        # Save result to file
        output_file = Path(self.folder_name) / f"{self.workflow_name}_result.json"
        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(result, indent=2))

        logger.info(f"Completed {self.workflow_name}, saved to {output_file}")
        return result
