import asyncio
import time
from typing import List, Dict, Any, AsyncGenerator

from ..types import LLMMessage, StreamEvent, TokenUsage, ToolUseContext, LLMAdapter, LLMChatOptions
from ..tool.framework import ToolRegistry
from ..tool.executor import ToolExecutor

class RunnerOptions:
    def __init__(self, model: str, systemPrompt: str = None, maxTurns: int = 10, maxTokens: int = None, temperature: float = None, allowedTools: List[str] = None, agentName: str = None, agentRole: str = None):
        self.model = model
        self.systemPrompt = systemPrompt
        self.maxTurns = maxTurns
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.allowedTools = allowedTools
        self.agentName = agentName
        self.agentRole = agentRole

class AgentRunner:
    def __init__(self, adapter: LLMAdapter, tool_registry: ToolRegistry, tool_executor: ToolExecutor, options: RunnerOptions):
        self.adapter = adapter
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.options = options
        self.maxTurns = options.maxTurns or 10

    async def run(self, messages: List[LLMMessage], onMessage=None, onToolCall=None, onToolResult=None) -> Dict[str, Any]:
        accumulated = {
            "messages": [], "output": "", "toolCalls": [], "tokenUsage": {"input_tokens":0, "output_tokens":0}, "turns": 0
        }
        async for event in self.stream(messages, onMessage, onToolCall, onToolResult):
            if event['type'] == 'done':
                accumulated.update(event['data'])
            elif event['type'] == 'error':
                raise event['data']
        return accumulated

    async def stream(self, initial_messages: List[LLMMessage], onMessage=None, onToolCall=None, onToolResult=None) -> AsyncGenerator[StreamEvent, None]:
        conversation = list(initial_messages)
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        all_tool_calls = []
        final_output = ""
        turns = 0

        all_defs = self.tool_registry.to_tool_defs()
        if self.options.allowedTools:
            tool_defs = [d for d in all_defs if d['name'] in self.options.allowedTools]
        else:
            tool_defs = all_defs

        base_options = {
            "model": self.options.model,
            "tools": tool_defs if tool_defs else None,
            "maxTokens": self.options.maxTokens,
            "temperature": self.options.temperature,
            "systemPrompt": self.options.systemPrompt,
        }

        try:
            while turns < self.maxTurns:
                turns += 1

                response_stream = self.adapter.stream(conversation, base_options)
                
                turn_text = ""
                tool_use_blocks = []
                
                async for event in response_stream:
                    if event['type'] == 'text':
                        turn_text += event['data']
                        yield event
                    elif event['type'] == 'tool_use':
                        tool_use_blocks.append(event['data'])
                        yield event
                    elif event['type'] == 'done':
                        resp = event['data']
                        total_usage['input_tokens'] += resp['usage']['input_tokens']
                        total_usage['output_tokens'] += resp['usage']['output_tokens']
                    elif event['type'] == 'error':
                        raise event['data']

                assistant_msg = {"role": "assistant", "content": []}
                if turn_text:
                    assistant_msg['content'].append({"type": "text", "text": turn_text})
                assistant_msg['content'].extend(tool_use_blocks)

                conversation.append(assistant_msg)
                if onMessage:
                    onMessage(assistant_msg)

                if not tool_use_blocks:
                    final_output = turn_text
                    break

                tool_context: ToolUseContext = {
                    "agent": {
                        "name": self.options.agentName or "runner",
                        "role": self.options.agentRole or "assistant",
                        "model": self.options.model
                    },
                    "team": None,
                    "abort_event": None,
                    "cwd": None,
                    "metadata": None
                }

                async def exec_tool(block):
                    if onToolCall: onToolCall(block['name'], block['input'])
                    start_t = time.time()
                    res = await self.tool_executor.execute(block['name'], block['input'], tool_context)
                    dur = int((time.time() - start_t) * 1000)
                    if onToolResult: onToolResult(block['name'], res)
                    
                    record = {"toolName": block['name'], "input": block['input'], "output": res['data'], "duration": dur}
                    res_block = {"type": "tool_result", "tool_use_id": block['id'], "content": res['data'], "is_error": res.get('isError', False)}
                    return res_block, record

                executions = await asyncio.gather(*(exec_tool(b) for b in tool_use_blocks))
                
                tool_result_blocks = []
                for res_block, record in executions:
                    all_tool_calls.append(record)
                    tool_result_blocks.append(res_block)
                    yield {"type": "tool_result", "data": res_block}

                tool_result_msg = {"role": "user", "content": tool_result_blocks}
                conversation.append(tool_result_msg)
                if onMessage:
                    onMessage(tool_result_msg)

        except Exception as e:
            yield {"type": "error", "data": e}
            return

        if not final_output and conversation:
            for msg in reversed(conversation):
                if msg['role'] == 'assistant':
                    final_output = "".join([b['text'] for b in msg['content'] if b['type'] == 'text'])
                    break

        yield {
            "type": "done",
            "data": {
                "messages": conversation[len(initial_messages):],
                "output": final_output,
                "toolCalls": all_tool_calls,
                "tokenUsage": total_usage,
                "turns": turns
            }
        }
