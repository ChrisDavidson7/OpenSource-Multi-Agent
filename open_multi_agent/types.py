from typing import Any, Dict, List, Literal, Optional, Protocol, Union
from typing_extensions import TypedDict
from datetime import datetime
import asyncio

# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

class TextBlock(TypedDict):
    type: Literal['text']
    text: str

class ToolUseBlock(TypedDict):
    type: Literal['tool_use']
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultBlock(TypedDict):
    type: Literal['tool_result']
    tool_use_id: str
    content: str
    is_error: Optional[bool]

class ImageSource(TypedDict):
    type: Literal['base64']
    media_type: str
    data: str

class ImageBlock(TypedDict):
    type: Literal['image']
    source: ImageSource

ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]

# ---------------------------------------------------------------------------
# LLM messages & responses
# ---------------------------------------------------------------------------

class LLMMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: List[ContentBlock]

class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int

class LLMResponse(TypedDict):
    id: str
    content: List[ContentBlock]
    model: str
    stop_reason: str
    usage: TokenUsage

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class StreamEvent(TypedDict):
    type: Literal['text', 'tool_use', 'tool_result', 'done', 'error']
    data: Any

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class LLMToolDef(TypedDict):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class AgentInfo(TypedDict):
    name: str
    role: str
    model: str

class ToolUseContext(TypedDict):
    agent: AgentInfo
    team: Optional[Any] # TeamInfo
    abort_event: Optional[asyncio.Event]
    cwd: Optional[str]
    metadata: Optional[Dict[str, Any]]

class ToolResult(TypedDict):
    data: str
    isError: Optional[bool]

class ToolDefinition(Protocol):
    name: str
    description: str
    inputSchema: Any # Pydantic BaseModel Type
    
    async def execute(self, input: Any, context: ToolUseContext) -> ToolResult:
        ...

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentConfig(TypedDict):
    name: str
    model: str
    provider: Optional[Literal['anthropic', 'openai']]
    systemPrompt: Optional[str]
    tools: Optional[List[str]]
    maxTurns: Optional[int]
    maxTokens: Optional[int]
    temperature: Optional[float]

class AgentState(TypedDict):
    status: Literal['idle', 'running', 'completed', 'error']
    messages: List[LLMMessage]
    tokenUsage: TokenUsage
    error: Optional[Exception]

class ToolCallRecord(TypedDict):
    toolName: str
    input: Dict[str, Any]
    output: str
    duration: int

class AgentRunResult(TypedDict):
    success: bool
    output: str
    messages: List[LLMMessage]
    tokenUsage: TokenUsage
    toolCalls: List[ToolCallRecord]

# ---------------------------------------------------------------------------
# Team & Task
# ---------------------------------------------------------------------------

class TeamConfig(TypedDict):
    name: str
    agents: List[AgentConfig]
    sharedMemory: Optional[bool]
    maxConcurrency: Optional[int]

class TeamRunResult(TypedDict):
    success: bool
    agentResults: Dict[str, AgentRunResult]
    totalTokenUsage: TokenUsage

TaskStatus = Literal['pending', 'in_progress', 'completed', 'failed', 'blocked']

class Task(TypedDict):
    id: str
    title: str
    description: str
    status: TaskStatus
    assignee: Optional[str]
    dependsOn: Optional[List[str]]
    result: Optional[str]
    createdAt: datetime
    updatedAt: datetime

# ---------------------------------------------------------------------------
# Orchestrator & Memory
# ---------------------------------------------------------------------------

class OrchestratorEvent(TypedDict):
    type: Literal['agent_start', 'agent_complete', 'task_start', 'task_complete', 'message', 'error']
    agent: Optional[str]
    task: Optional[str]
    data: Optional[Any]

class OrchestratorConfig(TypedDict):
    maxConcurrency: Optional[int]
    defaultModel: Optional[str]
    defaultProvider: Optional[Literal['anthropic', 'openai']]
    onProgress: Optional[Any] # Callable[[OrchestratorEvent], None]

class MemoryEntry(TypedDict):
    key: str
    value: str
    metadata: Optional[Dict[str, Any]]
    createdAt: datetime

class MemoryStore(Protocol):
    async def get(self, key: str) -> Optional[MemoryEntry]: ...
    async def set(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> None: ...
    async def list(self) -> List[MemoryEntry]: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...

# ---------------------------------------------------------------------------
# LLM Adapter
# ---------------------------------------------------------------------------

from typing import AsyncGenerator

class LLMChatOptions(TypedDict):
    model: str
    tools: Optional[List[LLMToolDef]]
    maxTokens: Optional[int]
    temperature: Optional[float]
    systemPrompt: Optional[str]
    abort_event: Optional[asyncio.Event]

class LLMStreamOptions(LLMChatOptions):
    pass

class LLMAdapter(Protocol):
    name: str

    async def chat(self, messages: List[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        ...

    async def stream(self, messages: List[LLMMessage], options: LLMStreamOptions) -> AsyncGenerator[StreamEvent, None]:
        ...
        yield # To satisfy linters
