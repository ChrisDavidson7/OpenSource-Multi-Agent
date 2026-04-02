import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from open_multi_agent import OpenMultiAgent

def on_progress(event):
    if event['type'] == 'agent_start':
        print(f"\n[{event['agent']}] is starting their assigned task...")
    elif event['type'] == 'task_complete':
        print(f"[{event['agent']}] successfully finished their task!")
    elif event['type'] == 'error':
        print(f"ERROR from [{event.get('agent', 'system')}]:", event.get('data'))

async def main():
    orchestrator = OpenMultiAgent({
        "defaultModel": 'gpt-5-mini',
        "defaultProvider": 'openai',
        "onProgress": on_progress
    })
    
    architect = {
        "name": 'architect',
        "model": 'gpt-5-mini',
        "provider": 'openai',
        "systemPrompt": 'You design clean technical specifications. You do not write source code, you only write markdown plans.',
        "tools": ['file_write'],
    }
    
    developer = {
        "name": 'developer',
        "model": 'gpt-5-mini',
        "provider": 'openai',
        "systemPrompt": 'You strictly implement working python code based on the architects plan. Write clean python code.',
        "tools": ['bash', 'file_read', 'file_write'],
    }
    
    qa = {
        "name": 'qa-tester',
        "model": 'gpt-5-mini',
        "provider": 'openai',
        "systemPrompt": 'You are quality assurance. You run scripts safely using bash to verify they work exactly as required without throwing exceptions.',
        "tools": ['bash', 'file_read'],
    }

    print('Assembling the Native Python Software Team (Architect, Developer, QA)...')
    team = orchestrator.createTeam('software-team', {
        "name": 'software-team',
        "agents": [architect, developer, qa],
        "sharedMemory": True,
    })

    print('\nGiving the team a high-level goal...')
    goal = 'Create a simple Python CLI script named `hello_team_python.py` in the current directory that prints "Hello, Native Python Multi-Agent World!". Run it to verify it works.'
    
    result = await orchestrator.runTeam(team, goal)

    print('\n=============================================')
    print('TEAM EXECUTION FINISHED')
    print('Success:', result['success'])
    print('Total Tokens Used:', result['totalTokenUsage'])
    
    print('\nSynthesized Final Report:')
    print(result['agentResults'].get('coordinator', {}).get('output'))

if __name__ == "__main__":
    asyncio.run(main())
