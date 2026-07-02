from __future__ import annotations

from typing import Any, Dict


class MCPService:
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "plan_goal":
            goal = arguments.get("goal", "")
            return {
                "status": "ok",
                "tool_name": tool_name,
                "result": f"Planned workflow for {goal}",
            }

        return {
            "status": "error",
            "tool_name": tool_name,
            "result": "Unknown tool",
        }
