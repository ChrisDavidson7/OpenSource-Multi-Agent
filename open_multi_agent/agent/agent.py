from typing import Any, AsyncGenerator, List, Dict
from ..types import AgentConfig, AgentState, AgentRunResult, LLMMessage, StreamEvent, ToolUseContext
from ..tool.framework import ToolRegistry, ToolDefinition
from ..tool.executor import ToolExecutor
from ..llm.adapter import create_adapter
from .runner import AgentRunner, RunnerOptions

class Agent:
    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry, tool_executor: ToolExecutor):
        self.name = config['name']
        self.config = config
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self.state: AgentState = {"status": "idle", "messages": [], "tokenUsage": {"input_tokens":0, "output_tokens":0}, "error": None}
        self.message_history: List[LLMMessage] = []
        self._runner = None

    async def _get_runner(self) -> AgentRunner:
        if self._runner:
            return self._runner
        
        provider = self.config.get('provider', 'anthropic')
        adapter = await create_adapter(provider)
        
        options = RunnerOptions(
            model=self.config['model'],
            systemPrompt=self.config.get('systemPrompt'),
            maxTurns=self.config.get('maxTurns', 10),
            maxTokens=self.config.get('maxTokens'),
            temperature=self.config.get('temperature'),
            allowedTools=self.config.get('tools'),
            agentName=self.name,
            agentRole=self.config.get('systemPrompt', '')[:50]
        )
        self._runner = AgentRunner(adapter, self._tool_registry, self._tool_executor, options)
        return self._runner

    async def run(self, prompt: str) -> AgentRunResult:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return await self._execute_run(messages)

    async def prompt(self, message: str) -> AgentRunResult:
        user_message = {"role": "user", "content": [{"type": "text", "text": message}]}
        self.message_history.append(user_message)
        result = await self._execute_run(list(self.message_history))
        self.message_history.extend(result['messages'])
        return result

    async def stream(self, prompt: str) -> AsyncGenerator[StreamEvent, None]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        async for ev in self._execute_stream(messages):
            yield ev

    def reset(self):
        self.message_history = []
        self.state = {"status": "idle", "messages": [], "tokenUsage": {"input_tokens":0, "output_tokens":0}, "error": None}

    def add_tool(self, tool: ToolDefinition):
        self._tool_registry.register(tool)

    async def _execute_run(self, messages: List[LLMMessage]) -> AgentRunResult:
        self.state['status'] = 'running'
        try:
            runner = await self._get_runner()
            def on_msg(m): self.state['messages'].append(m)
            
            res = await runner.run(messages, onMessage=on_msg)
            
            self.state['tokenUsage']['input_tokens'] += res['tokenUsage']['input_tokens']
            self.state['tokenUsage']['output_tokens'] += res['tokenUsage']['output_tokens']
            self.state['status'] = 'completed'
            
            return {
                "success": True,
                "output": res['output'],
                "messages": res['messages'],
                "tokenUsage": res['tokenUsage'],
                "toolCalls": res['toolCalls']
            }
        except Exception as e:
            self.state['status'] = 'error'
            self.state['error'] = e
            return {
                "success": False,
                "output": str(e),
                "messages": [],
                "tokenUsage": {"input_tokens":0, "output_tokens":0},
                "toolCalls": []
            }

    async def _execute_stream(self, messages: List[LLMMessage]) -> AsyncGenerator[StreamEvent, None]:
        self.state['status'] = 'running'
        try:
            runner = await self._get_runner()
            async for event in runner.stream(messages):
                if event['type'] == 'done':
                    res = event['data']
                    self.state['tokenUsage']['input_tokens'] += res['tokenUsage']['input_tokens']
                    self.state['tokenUsage']['output_tokens'] += res['tokenUsage']['output_tokens']
                    self.state['status'] = 'completed'
                elif event['type'] == 'error':
                    self.state['status'] = 'error'
                    self.state['error'] = event['data']
                yield event
        except Exception as e:
            self.state['status'] = 'error'
            self.state['error'] = e
            yield {"type": "error", "data": e}
