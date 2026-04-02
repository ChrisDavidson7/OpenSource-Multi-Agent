from typing import List
from ..task.queue import TaskQueue

class Scheduler:
    def __init__(self, strategy: str = 'dependency-first'):
        self.strategy = strategy

    def autoAssign(self, queue: TaskQueue, agents: List[dict]):
        for task in queue.tasks.values():
            if not task.get('assignee') and task['status'] not in ('completed', 'failed'):
                if len(agents) > 0:
                    task['assignee'] = agents[0]['name']
