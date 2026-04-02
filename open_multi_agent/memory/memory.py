import datetime
from typing import List, Dict, Any, Optional
from ..types import MemoryEntry, MemoryStore

class SharedMemory(MemoryStore):
    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> Optional[MemoryEntry]:
        return self.entries.get(key)

    async def set(self, key: str, value: str, metadata: Dict[str, Any] = None) -> None:
        self.entries[key] = {
            "key": key,
            "value": value,
            "metadata": metadata,
            "createdAt": datetime.datetime.now()
        }

    async def list(self) -> List[MemoryEntry]:
        return list(self.entries.values())

    async def delete(self, key: str) -> None:
        self.entries.pop(key, None)

    async def clear(self) -> None:
        self.entries.clear()

    async def get_summary(self) -> str:
        all_entries = await self.list()
        if not all_entries:
            return ""
        lines = ["## Shared Context (from previous agents)"]
        for e in all_entries:
            # Simple summarization strategy mirroring TS
            lines.append(f"### Output ({e['key']})\n{e['value']}")
        return "\n".join(lines)
