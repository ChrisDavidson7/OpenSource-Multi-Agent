import json
import re
import asyncio
from typing import Dict, Any, List

from ..types import OrchestratorConfig, OrchestratorEvent, AgentConfig, AgentRunResult, TeamRunResult, Task
from ..agent.agent import Agent
from ..agent.pool import AgentPool
from ..tool.framework import ToolRegistry
from ..tool.executor import ToolExecutor
from ..tool.built_in import register_built_in_tools
from ..team.team import Team
from ..task.task import create_task
from ..task.queue import TaskQueue
from .scheduler import Scheduler

def build_agent(config: AgentConfig) -> Agent:
    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)
    return Agent(config, registry, executor)

def parse_task_specs(raw: str) -> List[dict]:
    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    candidate = match.group(1) if match else raw
    start = candidate.find('[')
    end = candidate.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start:end+1])
        if not isinstance(parsed, list): return None
        return parsed
    except:
        return None

class RunContext:
    def __init__(self, team, pool, scheduler, config, agent_results):
        self.team = team
        self.pool = pool
        self.scheduler = scheduler
        self.config = config
        self.agent_results = agent_results

async def execute_queue(queue: TaskQueue, ctx: RunContext):
    pool = ctx.pool
    team = ctx.team
    config = ctx.config

    while True:
        ctx.scheduler.autoAssign(queue, team.get_agents())
        pending = queue.get_by_status('pending')
        if not pending:
            break

        async def run_task(task: Task):
            queue.update(task['id'], {"status": "in_progress"})
            assignee = task['assignee']
            
            if config.get('onProgress'):
                config['onProgress']({"type": "task_start", "task": task['id'], "agent": assignee})
                config['onProgress']({"type": "agent_start", "agent": assignee, "task": task['id']})

            prompt_lines = [f"# Task: {task['title']}\n\n{task['description']}"]
            
            sm = team.get_shared_memory_instance()
            if sm:
                summary = await sm.get_summary()
                if summary: prompt_lines.extend(["", summary])
                
            msgs = team.get_messages(assignee)
            if msgs:
                prompt_lines.extend(["", "## Messages"])
                for m in msgs: prompt_lines.append(f"- **{m['from']}**: {m['content']}")

            try:
                res = await pool.run(assignee, "\n".join(prompt_lines))
                ctx.agent_results[f"{assignee}:{task['id']}"] = res
                if res['success']:
                    if sm: await sm.set(f"task:{task['id']}:result", res['output'], {"agent": assignee})
                    queue.complete(task['id'], res['output'])
                    if config.get('onProgress'):
                        config['onProgress']({"type": "agent_complete", "agent": assignee, "task": task['id'], "data": res})
                        config['onProgress']({"type": "task_complete", "task": task['id'], "agent": assignee, "data": res})
                else:
                    queue.fail(task['id'], res['output'])
                    if config.get('onProgress'):
                        config['onProgress']({"type": "error", "agent": assignee, "task": task['id'], "data": res['output']})
            except Exception as e:
                queue.fail(task['id'], str(e))
                if config.get('onProgress'):
                    config['onProgress']({"type": "error", "agent": assignee, "task": task['id'], "data": str(e)})

        await asyncio.gather(*(run_task(t) for t in pending))

class OpenMultiAgent:
    def __init__(self, config: OrchestratorConfig = None):
        config = config or {}
        self.config = {
            "maxConcurrency": config.get("maxConcurrency", 5),
            "defaultModel": config.get("defaultModel", "gpt-4o"),
            "defaultProvider": config.get("defaultProvider", "openai"),
            "onProgress": config.get("onProgress")
        }
        self.teams: Dict[str, Team] = {}

    def createTeam(self, name: str, config: Dict[str, Any]) -> Team:
        team = Team(config)
        self.teams[name] = team
        return team

    async def runAgent(self, config: AgentConfig, prompt: str) -> AgentRunResult:
        agent = build_agent(config)
        if self.config.get('onProgress'):
            self.config['onProgress']({"type": "agent_start", "agent": config['name']})
        
        result = await agent.run(prompt)
        
        if self.config.get('onProgress'):
            self.config['onProgress']({"type": "agent_complete", "agent": config['name']})
        return result

    async def runTeam(self, team: Team, goal: str) -> TeamRunResult:
        agents = team.get_agents()
        names = ", ".join(a['name'] for a in agents)
        roster = "\n".join(f"- **{a['name']}** ({a.get('model')}): {a.get('systemPrompt', '')[:120]}" for a in agents)
        
        sys_prompt = f"""You are a task coordinator. Decompose goals into tasks for:
## Roster
{roster}

Respond ONLY with a JSON array wrapped in a ```json code fence. Each task:
  - "title": Title
  - "description": Description
  - "assignee": Assignee name
  - "dependsOn": Array of title strings this depends on"""

        coord_cfg = {
            "name": "coordinator",
            "model": self.config['defaultModel'],
            "provider": self.config['defaultProvider'],
            "systemPrompt": sys_prompt,
            "maxTurns": 3
        }

        coordinator = build_agent(coord_cfg)
        
        if self.config.get('onProgress'):
            self.config['onProgress']({"type": "agent_start", "agent": "coordinator"})
            
        decomp_prompt = f"Decompose this goal into tasks for ({names}):\n\n{goal}"
        decomp_res = await coordinator.run(decomp_prompt)
        
        specs = parse_task_specs(decomp_res['output'])
        queue = TaskQueue()
        agent_names = {a['name'] for a in agents}

        if specs:
            tasks = []
            title_id_map = {}
            for s in specs:
                t = create_task(s.get('title', 'Task'), s.get('description', ''), s.get('assignee') if s.get('assignee') in agent_names else None)
                title_id_map[s.get('title', '').lower()] = t['id']
                tasks.append((t, s.get('dependsOn', [])))
            
            for t, deps in tasks:
                real_deps = [title_id_map[d.lower()] for d in deps if d.lower() in title_id_map]
                t['dependsOn'] = real_deps
                queue.add(t)
        else:
            for a in agents:
                queue.add(create_task(f"{a['name']}: {goal[:30]}", goal, a['name']))

        scheduler = Scheduler()
        pool = AgentPool(self.config['maxConcurrency'])
        for a in agents:
            pool.add(build_agent(a))
            
        ctx = RunContext(team, pool, scheduler, self.config, {"coordinator:decompose": decomp_res})
        
        await execute_queue(queue, ctx)
        
        # Synthesis
        comp = queue.get_by_status('completed')
        failed = queue.get_by_status('failed')
        
        syn_prompt = f"## Goal\n{goal}\n\n## Tasks\n"
        for t in comp:
            syn_prompt += f"### {t['title']} ({t['assignee']})\n{t['result']}\n\n"
        for t in failed:
            syn_prompt += f"### FAILED {t['title']} ({t['assignee']})\n{t['result']}\n\n"
            
        syn_res = await coordinator.run("Synthesize this into a final answer:\n" + syn_prompt)
        ctx.agent_results['coordinator'] = syn_res
        
        if self.config.get('onProgress'):
            self.config['onProgress']({"type": "agent_complete", "agent": "coordinator"})

        tot_in = sum(r.get('tokenUsage', {}).get('input_tokens', 0) for r in ctx.agent_results.values())
        tot_out = sum(r.get('tokenUsage', {}).get('output_tokens', 0) for r in ctx.agent_results.values())

        return {
            "success": len(failed) == 0,
            "agentResults": ctx.agent_results,
            "totalTokenUsage": {"input_tokens": tot_in, "output_tokens": tot_out}
        }
