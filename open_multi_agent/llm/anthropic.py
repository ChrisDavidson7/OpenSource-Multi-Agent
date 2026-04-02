import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from anthropic import AsyncAnthropic

from ..types import LLMMessage, LLMChatOptions, LLMStreamOptions, LLMResponse, StreamEvent

def _to_anthropic_tools(tools):
    if not tools:
        return None
    res = []
    for t in tools:
        res.append({
            "name": t['name'],
            "description": t.get('description', ''),
            "input_schema": t['inputSchema']
        })
    return res

class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, api_key: str = None):
        self.client = AsyncAnthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'))

    async def chat(self, messages: List[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        kwargs = {
            "model": options['model'],
            "messages": messages,
            "max_tokens": options.get('maxTokens', 4096)
        }
        if options.get('systemPrompt'):
            kwargs['system'] = options['systemPrompt']
        if options.get('temperature') is not None:
            kwargs['temperature'] = options['temperature']
        if options.get('tools'):
            kwargs['tools'] = _to_anthropic_tools(options['tools'])

        response = await self.client.messages.create(**kwargs)
        
        content = []
        for block in response.content:
            if block.type == 'text':
                content.append({"type": "text", "text": block.text})
            elif block.type == 'tool_use':
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        return {
            "id": response.id,
            "content": content,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": getattr(response.usage, 'input_tokens', 0),
                "output_tokens": getattr(response.usage, 'output_tokens', 0)
            }
        }

    async def stream(self, messages: List[LLMMessage], options: LLMStreamOptions) -> AsyncGenerator[StreamEvent, None]:
        kwargs = {
            "model": options['model'],
            "messages": messages,
            "max_tokens": options.get('maxTokens', 4096),
            "stream": True
        }
        if options.get('systemPrompt'):
            kwargs['system'] = options['systemPrompt']
        if options.get('temperature') is not None:
            kwargs['temperature'] = options['temperature']
        if options.get('tools'):
            kwargs['tools'] = _to_anthropic_tools(options['tools'])

        try:
            stream = await self.client.messages.create(**kwargs)
            
            comp_id = ""
            comp_model = options['model']
            finish_reason = "end_turn"
            in_tokens = 0
            out_tokens = 0
            
            content_blocks = []
            current_text = ""
            current_tool = None

            async for event in stream:
                if event.type == 'message_start':
                    comp_id = event.message.id
                    comp_model = event.message.model
                    in_tokens = getattr(event.message.usage, 'input_tokens', 0)
                    
                elif event.type == 'content_block_start':
                    block = event.content_block
                    if block.type == 'tool_use':
                        current_tool = {"type": "tool_use", "id": block.id, "name": block.name, "input_str": ""}
                
                elif event.type == 'content_block_delta':
                    delta = event.delta
                    if delta.type == 'text_delta':
                        current_text += delta.text
                        yield {"type": "text", "data": delta.text}
                    elif delta.type == 'input_json_delta':
                        current_tool['input_str'] += delta.partial_json
                
                elif event.type == 'content_block_stop':
                    if current_tool:
                        try:
                            current_tool['input'] = json.loads(current_tool['input_str'])
                        except json.JSONDecodeError:
                            current_tool['input'] = {}
                        del current_tool['input_str']
                        content_blocks.append(current_tool)
                        yield {"type": "tool_use", "data": current_tool}
                        current_tool = None
                    elif current_text:
                        content_blocks.append({"type": "text", "text": current_text})
                        current_text = ""
                
                elif event.type == 'message_delta':
                    finish_reason = event.delta.stop_reason
                    out_tokens = getattr(event.usage, 'output_tokens', 0)

            if finish_reason == 'stop_sequence':
                finish_reason = 'end_turn'
                
            yield {
                "type": "done",
                "data": {
                    "id": comp_id,
                    "content": content_blocks,
                    "model": comp_model,
                    "stop_reason": finish_reason,
                    "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}
                }
            }
        except Exception as e:
            yield {"type": "error", "data": e}
