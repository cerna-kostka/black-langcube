"""
Dummy Graph1 implementation for Black LangCube examples.
This replaces the complex graph1.py with a simple mock implementation.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from black_langcube.graf.graph_base import BaseGraph

logger = logging.getLogger(__name__)


class Graph1(BaseGraph):
    """Mock Graph1 - Question processing and language detection."""
    
    def __init__(self, user_message: str, folder_name: str, language: Optional[str] = None):
        # Import GraphState here to avoid circular imports
        from black_langcube.graf.graph_base import GraphState
        super().__init__(GraphState, user_message, folder_name, language)
    
    @property
    def workflow_name(self):
        return "question_processing"
    
    def run(self) -> Dict[str, Any]:
        """Mock question processing workflow."""
        logger.info("Running Graph1 - Question processing and language detection")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Create output folder if it doesn't exist
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        
        # Generate mock result
        result = {
            "input_message": self.user_message,
            "language": self.language,
            "workflow": self.workflow_name,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mock_analysis": f"Analyzed '{self.user_message[:30]}...' using {self.workflow_name}",
            "tokens_used": len(self.user_message.split()) * 10,  # Mock token count
            "confidence": 0.85,
            "detected_language": self.language,
            "question_type": "informational" if "what" in self.user_message.lower() else "analytical",
            "entities": ["AI", "healthcare", "technology"],  # Mock entities
            "complexity_score": 0.7,
            "processing_nodes": ["language_detector", "entity_extractor", "question_classifier"]
        }
        
        # Save result to file
        output_file = Path(self.folder_name) / f"{self.workflow_name}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed {self.workflow_name}, saved to {output_file}")
        return result
