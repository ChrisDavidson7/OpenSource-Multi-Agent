import datetime
import uuid
from typing import List
from ..types import Task

def create_task(title: str, description: str, assignee: str = None, dependsOn: List[str] = None) -> Task:
    now = datetime.datetime.now()
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "status": "pending",
        "assignee": assignee,
        "dependsOn": dependsOn or [],
        "result": None,
        "createdAt": now,
        "updatedAt": now
    }
