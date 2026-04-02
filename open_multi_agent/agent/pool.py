import asyncio
from typing import Dict, Any
from .agent import Agent

class AgentPool:
    def __init__(self, max_concurrency: int):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.agents: Dict[str, Agent] = {}

    def add(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        return self.agents.get(name)

    async def run(self, name: str, prompt: str) -> Dict[str, Any]:
        agent = self.get(name)
        if not agent:
            raise ValueError(f"Agent {name} not found in pool")
        
        async with self.semaphore:
            return await agent.run(prompt)
