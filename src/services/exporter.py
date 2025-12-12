"""Export services for simulation results.

Handles saving strategies to JSON files for persistence and analysis.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.logging import get_logger

log = get_logger(__name__)


class ResultExporter:
    """Handles exporting of simulation results."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize exporter.
        
        Args:
            output_dir: Directory where results will be saved.
        """
        self.output_dir = output_dir
        self._ensure_dir()
        
    def _ensure_dir(self):
        """Ensure output directory exists."""
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
                log.info("created_output_directory", path=self.output_dir)
            except Exception as e:
                log.error("output_directory_creation_failed", error=str(e))

    def save_results(
        self, 
        strategies: List[Dict[str, Any]], 
        prefix: str = "simulation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save results to a JSON file.
        
        Args:
            strategies: List of strategy dictionaries.
            prefix: Filename prefix.
            metadata: Optional metadata to include in the file (e.g. config used).
            
        Returns:
            Path to the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "count": len(strategies),
                **(metadata or {})
            },
            "strategies": strategies
        }
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            
            log.info("results_saved", path=filepath, count=len(strategies))
            return filepath
            
        except Exception as e:
            log.error("results_save_failed", path=filepath, error=str(e))
            raise
