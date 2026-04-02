from typing import List, Dict
import datetime
from ..types import Task

class TaskQueue:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add(self, task: Task):
        self.tasks[task['id']] = task
        self._update_blocked_states()

    def list(self) -> List[Task]:
        return list(self.tasks.values())

    def get(self, task_id: str) -> Task:
        return self.tasks.get(task_id)

    def get_by_status(self, status: str) -> List[Task]:
        ans = []
        for t in self.tasks.values():
            if t['status'] == status:
                # If asking for pending, ensure it's unblocked
                if status == 'pending':
                    self._update_blocked_states() # Just in case
                    if t['status'] == 'pending':
                        ans.append(t)
                else:
                    ans.append(t)
        return ans

    def update(self, task_id: str, updates: dict):
        task = self.tasks[task_id]
        task.update(updates)
        task['updatedAt'] = datetime.datetime.now()
        if 'status' in updates:
            self._update_blocked_states()

    def complete(self, task_id: str, result: str):
        self.update(task_id, {"status": "completed", "result": result})

    def fail(self, task_id: str, error: str):
        self.update(task_id, {"status": "failed", "result": error})

    def _update_blocked_states(self):
        changed = True
        while changed:
            changed = False
            for t in list(self.tasks.values()):
                if t['status'] not in ('pending', 'blocked'):
                    continue
                
                # Check dependencies
                deps_status = [self.tasks[did]['status'] for did in t['dependsOn'] if did in self.tasks]
                
                if 'failed' in deps_status:
                    t['status'] = 'failed'
                    t['result'] = 'Cascade failure: dependency failed'
                    changed = True
                elif 'blocked' in deps_status or 'pending' in deps_status or 'in_progress' in deps_status:
                    if t['status'] != 'blocked':
                        t['status'] = 'blocked'
                        changed = True
                else:
                    # All completed
                    if t['status'] != 'pending':
                        t['status'] = 'pending'
                        changed = True
