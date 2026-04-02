# OpenSource Multi Agent

A powerful, native Python `asyncio` multi-agent orchestration framework. 

This framework empowers developers to build autonomous, multi-model AI teams that can mathematically decompose complex goals into parallel task graphs (DAGs) and securely execute them.

## Key Features

- **Automated Task Orchestration**: Provide a single high-level goal, and the framework spins up a "Coordinator" agent to generate a Directed Acyclic Graph (DAG) of tasks. 
- **Parallel Asynchronous Execution**: Utilizing native Python `asyncio`, independent tasks execution flows are parallelized optimally for speed while blocked dependencies safely yield.
- **Pydantic Tool Integration**: Write highly complex custom precision tools seamlessly utilizing `pydantic.BaseModel` to securely enforce LLM type constraints and schema logic.
- **Hybrid Multi-Model Support**: Native integrations to dynamically map both OpenAI (inclusive of Azure AI Foundry models as a service) and Anthropic models within the same team seamlessly!
- **Stateful Team Memory**: Agents share an internal memory bus, allowing downstream agents to read the validated outputs of their upstream partners inherently.
- **File System Arsenal**: Built-in, battle-tested `bash`, `file_edit`, `file_read`, and `rg`-backed `grep` tools mirroring the TypeScript standard.

## Installation

This project utilizes standard `pip`. Use Python 3.10+ for native `asyncio` features.

```bash
# Install the minimal dependencies
pip install -r requirements.txt
```

## Quick Start Example

The quickest way to see the magic is by running a multi-agent team! Before running, you must configure your environment variables for your chosen LLM provider.

### Option A: Standard OpenAI or Anthropic
If you are using the standard public APIs, simply export your key:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option B: Azure AI Foundry / Azure OpenAI
If you are using models hosted on Azure, this framework automatically reroutes traffic to your Azure endpoint natively. Export your Azure credentials like so:

**For Windows (PowerShell):**
```powershell
$env:AZURE_OPENAI_API_KEY="your_azure_api_key"
$env:AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"

# Required ONLY if using a Traditional Azure OpenAI endpoint (omit if using Foundry Serverless/MaaS endpoints with /v1/ in the path)
$env:OPENAI_API_VERSION="2024-02-15-preview"
```

**For Linux / macOS / Git Bash:**
```bash
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
```
*(The framework's `openai.py` adapter will automatically detect `AZURE_OPENAI_ENDPOINT`, smartly parse whether it's a traditional endpoint or a Foundry Serverless `/v1/` endpoint, and apply the correct headers/paths behind the scenes!)*

Once authenticated, run the built-in example which constructs an Architect, Developer, and QA tester working in tandem.

```bash
python examples/test_multi_agent.py
```

### The Architecture in Action

```python
import asyncio
from open_multi_agent import OpenMultiAgent

async def main():
    # 1. Initialize Orchestrator
    orchestrator = OpenMultiAgent({
        "defaultModel": 'gpt-4o',
        "defaultProvider": 'openai'
    })
    
    # 2. Define the team roster
    architect = {
        "name": 'architect',
        "model": 'gpt-4o',
        "provider": 'openai',
        "systemPrompt": 'You design technical specifications by writing markdown plans.',
        "tools": ['file_write']
    }
    
    developer = {
        "name": 'developer',
        "model": 'gpt-4o',
        "provider": 'openai',
        "systemPrompt": 'You implement working python code based on the architects plan.',
        "tools": ['bash', 'file_read', 'file_write']
    }

    # 3. Register the Team
    team = orchestrator.createTeam('software-team', {
        "name": 'software-team',
        "agents": [architect, developer],
        "sharedMemory": True,
    })

    # 4. Command the Team!
    goal = 'Design and write a python script called hello.py that solves the fibonacci sequence.'
    result = await orchestrator.runTeam(team, goal)
    
    print("Success:", result['success'])

if __name__ == "__main__":
    asyncio.run(main())
```

## Adding Custom Typed Tools

Define robust tools natively using Pydantic, making validation incredibly secure during LLM callback inferences:

```python
from open_multi_agent import define_tool
from pydantic import BaseModel

class MathInput(BaseModel):
    a: int
    b: int

async def add_numbers(input_data: MathInput, context):
    return {"data": str(input_data.a + input_data.b), "isError": False}

math_tool = define_tool(
    name="math_tool",
    description="Adds two numbers safely.",
    inputSchema=MathInput,
    execute=add_numbers
)
```

