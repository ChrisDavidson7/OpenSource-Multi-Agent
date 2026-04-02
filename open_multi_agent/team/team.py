from typing import List, Dict, Optional
from ..types import TeamConfig, AgentConfig
from ..memory.memory import SharedMemory

class Team:
    def __init__(self, config: TeamConfig):
        self.config = config
        self.name = config['name']
        self._shared_memory = SharedMemory() if config.get('sharedMemory') else None
        self._messages: Dict[str, List[dict]] = {}
    
    def get_agents(self) -> List[AgentConfig]:
        return self.config['agents']

    def get_shared_memory_instance(self) -> Optional[SharedMemory]:
        return self._shared_memory

    def get_messages(self, agent_name: str) -> List[dict]:
        return self._messages.get(agent_name, [])

    def add_message(self, to: str, from_agent: str, content: str):
        if to not in self._messages:
            self._messages[to] = []
        self._messages[to].append({"from": from_agent, "content": content})
