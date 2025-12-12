"""Export services for strategies and results.

JSON and CSV export functionality.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


class JSONExporter:
    """Export strategies and results to JSON format."""
    
    @staticmethod
    def export_strategies(
        strategies: List[Dict[str, Any]],
        include_simulation: bool = True,
    ) -> str:
        """Export strategies to JSON string.
        
        Args:
            strategies: List of strategy dicts
            include_simulation: Whether to include simulation data
            
        Returns:
            JSON string
        """
        from utils import to_json_safe
        
        data = to_json_safe(strategies)
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def export_to_file(
        strategies: List[Dict[str, Any]],
        filepath: str,
    ) -> None:
        """Export strategies to a JSON file.
        
        Args:
            strategies: List of strategy dicts
            filepath: Output file path
        """
        content = JSONExporter.export_strategies(strategies)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
