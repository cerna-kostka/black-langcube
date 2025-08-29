"""
Dummy Graph2 implementation for Black LangCube examples.
This provides a simple mock implementation for keyword processing.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from black_langcube.graf.graph_base import BaseGraph, GraphState

logger = logging.getLogger(__name__)


class Graph2(BaseGraph):
    """Mock Graph2 - Keyword processing."""
    
    def __init__(self, user_message: str, folder_name: str, language: Optional[str] = None):
        super().__init__(GraphState, user_message, folder_name, language)
    
    @property
    def workflow_name(self):
        return "keyword_processing"
    
    def run(self) -> Dict[str, Any]:
        """Mock keyword processing workflow."""
        logger.info("Running Graph2 - Keyword processing")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Create output folder if it doesn't exist
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        
        # Add Graph2-specific mock data
        words = self.user_message.lower().split()
        mock_keywords = [word for word in words if len(word) > 4][:5]  # Simple keyword extraction
        
        result = {
            "input_message": self.user_message,
            "language": self.language,
            "workflow": self.workflow_name,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mock_analysis": f"Analyzed '{self.user_message[:30]}...' using {self.workflow_name}",
            "tokens_used": len(self.user_message.split()) * 10,  # Mock token count
            "confidence": 0.85,
            "extracted_keywords": mock_keywords,
            "keyword_weights": {kw: round(len(kw) / 10, 2) for kw in mock_keywords},
            "semantic_clusters": ["technology", "healthcare", "research"],
            "processing_nodes": ["keyword_extractor", "semantic_analyzer", "weight_calculator"]
        }
        
        # Save result to file
        output_file = Path(self.folder_name) / f"{self.workflow_name}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed {self.workflow_name}, saved to {output_file}")
        return result
