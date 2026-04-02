import os
import re
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path

from ..types import ToolUseContext, ToolResult
from .framework import define_tool, ToolRegistry

# ----------------- BASH ----------------- #
class BashInput(BaseModel):
    command: str = Field(..., description="The shell command to execute.")
    timeout: Optional[int] = Field(120, description="Timeout in seconds.")

async def execute_bash(input_data: BashInput, context: ToolUseContext) -> ToolResult:
    cwd = context.get("cwd", os.getcwd())
    try:
        proc = await asyncio.create_subprocess_shell(
            input_data.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=input_data.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"data": f"Command timed out after {input_data.timeout} seconds.", "isError": True}

        output = stdout.decode() + stderr.decode()
        if proc.returncode != 0:
            return {"data": f"Command failed with exit code {proc.returncode}\n{output}", "isError": True}
        return {"data": output.strip() or "Command executed successfully with no output.", "isError": False}
    except Exception as e:
        return {"data": str(e), "isError": True}

bash_tool = define_tool(
    name="bash",
    description="Execute shell commands. Useful for running scripts, installing dependencies, or navigating the filesystem.",
    inputSchema=BashInput,
    execute=execute_bash
)

# ----------------- FILE READ ----------------- #
class FileReadInput(BaseModel):
    path: str = Field(..., description="Absolute path to the file to read.")
    offset: Optional[int] = Field(0, description="Line number to start reading from (0-indexed).")
    limit: Optional[int] = Field(None, description="Maximum number of lines to read.")

async def execute_file_read(input_data: FileReadInput, context: ToolUseContext) -> ToolResult:
    try:
        path = Path(input_data.path)
        if not path.is_absolute():
            cwd = context.get('cwd', os.getcwd())
            path = Path(cwd) / path
            
        if not path.exists() or not path.is_file():
            return {"data": f"File not found: {path}", "isError": True}

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = input_data.offset or 0
        end = start + input_data.limit if input_data.limit else len(lines)
        subset = lines[start:end]
        
        return {"data": "".join(subset), "isError": False}
    except Exception as e:
        return {"data": str(e), "isError": True}

file_read_tool = define_tool(
    name="file_read",
    description="Read file contents. Supports offset and limit for reading large files in chunks.",
    inputSchema=FileReadInput,
    execute=execute_file_read
)

# ----------------- FILE WRITE ----------------- #
class FileWriteInput(BaseModel):
    path: str = Field(..., description="Absolute path to write to.")
    content: str = Field(..., description="Full content to write to the file.")

async def execute_file_write(input_data: FileWriteInput, context: ToolUseContext) -> ToolResult:
    try:
        path = Path(input_data.path)
        if not path.is_absolute():
            path = Path(context.get('cwd', os.getcwd())) / path
            
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(input_data.content)
            
        return {"data": f"Successfully wrote to {path}", "isError": False}
    except Exception as e:
        return {"data": str(e), "isError": True}

file_write_tool = define_tool(
    name="file_write",
    description="Write or overwrite a file completely. Auto-creates parent directories.",
    inputSchema=FileWriteInput,
    execute=execute_file_write
)

file_write_tool = define_tool(
    name="file_write",
    description="Write or overwrite a file completely. Auto-creates parent directories.",
    inputSchema=FileWriteInput,
    execute=execute_file_write
)

# ----------------- FILE EDIT ----------------- #
class FileEditInput(BaseModel):
    path: str = Field(..., description="Absolute path to the file to edit.")
    old_string: str = Field(..., description="The exact string to find and replace. Must match character-for-character including whitespace and newlines.")
    new_string: str = Field(..., description="The replacement string that will be inserted in place of `old_string`.")
    replace_all: Optional[bool] = Field(False, description="When true, replace every occurrence instead of requiring uniqueness.")

async def execute_file_edit(input_data: FileEditInput, context: ToolUseContext) -> ToolResult:
    try:
        path = Path(input_data.path)
        if not path.is_absolute():
            path = Path(context.get('cwd', os.getcwd())) / path
            
        with open(path, 'r', encoding='utf-8') as f:
            original = f.read()
            
        occurrences = original.count(input_data.old_string)
        
        if occurrences == 0:
            return {"data": f"The string to replace was not found in {path}. Make sure it exactly matches.", "isError": True}
        
        if occurrences > 1 and not input_data.replace_all:
            return {"data": f"`old_string` appears {occurrences} times. Provide a more specific string or set replace_all=true.", "isError": True}
            
        updated = original.replace(input_data.old_string, input_data.new_string)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(updated)
            
        return {"data": f"Replaced {occurrences} occurrence(s) in {path}", "isError": False}
    except Exception as e:
        return {"data": str(e), "isError": True}

file_edit_tool = define_tool(
    name="file_edit",
    description="Edit a file by replacing a specific string with new content.",
    inputSchema=FileEditInput,
    execute=execute_file_edit
)

# ----------------- GREP ----------------- #
class GrepInput(BaseModel):
    pattern: str = Field(..., description="Regular expression pattern to search for in file contents.")
    path: Optional[str] = Field(None, description="Directory or file path to search in. Defaults to cwd.")
    glob: Optional[str] = Field(None, description="Glob pattern to filter which files are searched.")
    maxResults: Optional[int] = Field(100, description="Maximum number of matching lines to return.")

async def execute_grep(input_data: GrepInput, context: ToolUseContext) -> ToolResult:
    search_path = input_data.path or context.get('cwd', os.getcwd())
    max_results = input_data.maxResults or 100
    
    args = ['--line-number', '--no-heading', '--color=never', f"--max-count={max_results}"]
    if input_data.glob:
        args.extend(['--glob', input_data.glob])
    args.extend(['--', input_data.pattern, search_path])
    
    try:
        # We attempt ripping first, it's fast
        proc = await asyncio.create_subprocess_exec(
            'rg', *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=context.get('cwd', os.getcwd())
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode not in (0, 1):
            return {"data": f"ripgrep failed (exit {proc.returncode}): {stderr.decode()}", "isError": True}
            
        output = stdout.decode().strip()
        if not output:
            return {"data": "No matches found.", "isError": False}
            
        return {"data": output, "isError": False}
    except FileNotFoundError:
        # Note: We omit pure python recursive fallback here to keep it clean natively for the agent 
        # But this suffices for Linux/ripgrep equipped platforms.
        return {"data": "grep/rg not available in environment.", "isError": True}
    except Exception as e:
        return {"data": f"Error running grep: {e}", "isError": True}

grep_tool = define_tool(
    name="grep",
    description="Search for a regular-expression pattern in files utilizing ripgrep.",
    inputSchema=GrepInput,
    execute=execute_grep
)

def register_built_in_tools(registry: ToolRegistry):
    registry.register(bash_tool)
    registry.register(file_read_tool)
    registry.register(file_write_tool)
    registry.register(file_edit_tool)
    registry.register(grep_tool)
