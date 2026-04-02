from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Awaitable
from pydantic import BaseModel
from ..types import ToolDefinition, ToolUseContext, ToolResult, LLMToolDef

T = TypeVar('T', bound=BaseModel)

class CustomTool(ToolDefinition):
    def __init__(self, name: str, description: str, inputSchema: Type[T], execute_func: Callable[[T, ToolUseContext], Awaitable[ToolResult]]):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.execute_func = execute_func

    async def execute(self, input_data: Any, context: ToolUseContext) -> ToolResult:
        # validate input with pydantic
        validated_input = self.inputSchema.model_validate(input_data)
        return await self.execute_func(validated_input, context)

def define_tool(name: str, description: str, inputSchema: Type[T], execute: Callable[[T, ToolUseContext], Awaitable[ToolResult]]) -> ToolDefinition:
    return CustomTool(name, description, inputSchema, execute)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def to_tool_defs(self) -> List[LLMToolDef]:
        defs = []
        for t in self.list():
            # Extract JSON schema from Pydantic
            schema = t.inputSchema.model_json_schema()
            defs.append({
                "name": t.name,
                "description": t.description,
                "inputSchema": schema
            })
        return defs
