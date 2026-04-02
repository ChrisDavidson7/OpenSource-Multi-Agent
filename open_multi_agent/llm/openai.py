import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncOpenAI, AsyncAzureOpenAI

from ..types import LLMMessage, LLMChatOptions, LLMStreamOptions, LLMResponse, StreamEvent

def _to_openai_tools(tools):
    if not tools:
        return None
    res = []
    for t in tools:
        res.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"]
            }
        })
    return res

def _has_tool_results(msg: LLMMessage) -> bool:
    return any(b['type'] == 'tool_result' for b in msg['content'])

def _format_messages(messages: List[LLMMessage], system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    res = []
    if system_prompt:
        res.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        if msg['role'] == 'assistant':
            text_parts = []
            tool_calls = []
            for b in msg['content']:
                if b['type'] == 'text':
                    text_parts.append(b['text'])
                elif b['type'] == 'tool_use':
                    tool_calls.append({
                        "id": b['id'],
                        "type": "function",
                        "function": {
                            "name": b['name'],
                            "arguments": json.dumps(b['input'])
                        }
                    })
            if not text_parts and tool_calls:
                res.append({"role": "assistant", "tool_calls": tool_calls, "content": None})
            else:
                block = {"role": "assistant", "content": "".join(text_parts)}
                if tool_calls:
                    block["tool_calls"] = tool_calls
                res.append(block)
        else: # user
            if not _has_tool_results(msg):
                if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                    res.append({"role": "user", "content": msg['content'][0]['text']})
                else:
                    parts = []
                    for b in msg['content']:
                        if b['type'] == 'text':
                            parts.append({"type": "text", "text": b['text']})
                    res.append({"role": "user", "content": parts})
            else:
                # Splitting tool results to dedicated role='tool' messages
                non_tool = [b for b in msg['content'] if b['type'] != 'tool_result']
                if non_tool:
                    res.append({"role": "user", "content": [ {"type": "text", "text": b['text']} for b in non_tool if b['type'] == 'text' ]})
                
                for b in msg['content']:
                    if b['type'] == 'tool_result':
                        res.append({
                            "role": "tool",
                            "tool_call_id": b['tool_use_id'],
                            "content": b['content']
                        })
    return res

class OpenAIAdapter:
    name = "openai"

    def __init__(self, api_key: str = None):
        endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        if endpoint:
            if '/v1' in endpoint:
                key = api_key or os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
                self.client = AsyncOpenAI(
                    api_key=key,
                    base_url=endpoint.replace('/responses', '').replace('/responses/', ''),
                    default_headers={'api-key': key}
                )
            else:
                self.client = AsyncAzureOpenAI(
                    api_key=api_key or os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY'),
                    azure_endpoint=endpoint,
                    api_version=os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
                )
        else:
            self.client = AsyncOpenAI(api_key=api_key or os.environ.get('OPENAI_API_KEY'))

    async def chat(self, messages: List[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        fmt_msgs = _format_messages(messages, options.get('systemPrompt'))
        kwargs = {
            "model": options['model'],
            "messages": fmt_msgs,
            "stream": False
        }
        if options.get('maxTokens'):
            kwargs['max_tokens'] = options['maxTokens']
        if options.get('temperature') is not None:
            kwargs['temperature'] = options['temperature']
        if options.get('tools'):
            kwargs['tools'] = _to_openai_tools(options['tools'])

        response = await self.client.chat.completions.create(**kwargs)
        
        choice = response.choices[0]
        content = []
        if choice.message.content:
            content.append({"type": "text", "text": choice.message.content})
        
        for tc in (choice.message.tool_calls or []):
            try:
                parsedArgs = json.loads(tc.function.arguments)
            except:
                parsedArgs = {}
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": parsedArgs
            })

        return {
            "id": response.id,
            "content": content,
            "model": response.model,
            "stop_reason": choice.finish_reason if choice.finish_reason != "tool_calls" else "tool_use",
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        }

    async def stream(self, messages: List[LLMMessage], options: LLMStreamOptions) -> AsyncGenerator[StreamEvent, None]:
        fmt_msgs = _format_messages(messages, options.get('systemPrompt'))
        kwargs = {
            "model": options['model'],
            "messages": fmt_msgs,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        if options.get('maxTokens'):
            kwargs['max_tokens'] = options['maxTokens']
        if options.get('temperature') is not None:
            kwargs['temperature'] = options['temperature']
        if options.get('tools'):
            kwargs['tools'] = _to_openai_tools(options['tools'])

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            
            comp_id = ""
            comp_model = ""
            finish_reason = "stop"
            in_tokens = 0
            out_tokens = 0
            full_text = ""
            tool_buffers = {}
            
            async for chunk in stream:
                comp_id = chunk.id or comp_id
                comp_model = chunk.model or comp_model
                if getattr(chunk, 'usage', None):
                    in_tokens = chunk.usage.prompt_tokens
                    out_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    full_text += delta.content
                    yield {"type": "text", "data": delta.content}
                
                for tc in (delta.tool_calls or []):
                    idx = tc.index
                    if idx not in tool_buffers:
                        tool_buffers[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_buffers[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_buffers[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_buffers[idx]["args"] += tc.function.arguments
                        
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            final_tools = []
            for buf in tool_buffers.values():
                try:
                    args = json.loads(buf["args"])
                except:
                    args = {}
                block = {"type": "tool_use", "id": buf["id"], "name": buf["name"], "input": args}
                final_tools.append(block)
                yield {"type": "tool_use", "data": block}

            done_content = []
            if full_text:
                done_content.append({"type": "text", "text": full_text})
            done_content.extend(final_tools)

            if finish_reason == "stop":
                finish_reason = "end_turn"
            elif finish_reason == "tool_calls":
                finish_reason = "tool_use"

            yield {
                "type": "done",
                "data": {
                    "id": comp_id,
                    "content": done_content,
                    "model": comp_model,
                    "stop_reason": finish_reason,
                    "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}
                }
            }
        except Exception as e:
            yield {"type": "error", "data": e}
