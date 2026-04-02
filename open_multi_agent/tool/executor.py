from typing import Any, Dict
from ..types import ToolUseContext, ToolResult
from .framework import ToolRegistry

class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, name: str, input_data: Dict[str, Any], context: ToolUseContext) -> ToolResult:
        tool = self.registry.get(name)
        if not tool:
            return {
                "data": f"Error: Tool '{name}' is not registered or not available to this agent.",
                "isError": True
            }

        try:
            return await tool.execute(input_data, context)
        except Exception as e:
            return {
                "data": f"Error executing tool '{name}': {str(e)}",
                "isError": True
            }
