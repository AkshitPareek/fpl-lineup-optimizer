#!/usr/bin/env python3
"""
Parallel Agents Launcher

Launches 4 agents simultaneously to work on different tasks:
- Agent 1: Dashboard prediction integration
- Agent 2: EXP-033 research (position-specific models)
- Agent 3: Dashboard UI improvements
- Agent 4: Additional features (placeholder for user-defined tasks)

Usage:
    python parallel_agents_launcher.py
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


AGENTS = [
    {
        'id': 1,
        'name': 'Dashboard Predictions',
        'script': 'agents/agent1_dashboard_predictions.py',
        'description': 'Integrate EXP-032 predictions into dashboard',
        'estimated_time': '5 min'
    },
    {
        'id': 2,
        'name': 'EXP-033 Research',
        'script': 'agents/agent2_exp033_research.py',
        'description': 'Train position-specific models',
        'estimated_time': '10 min'
    },
    {
        'id': 3,
        'name': 'Dashboard UI',
        'script': 'agents/agent3_dashboard_ui.py',
        'description': 'Generate UI improvement components',
        'estimated_time': '3 min'
    },
    {
        'id': 4,
        'name': 'Feature Enhancement',
        'script': None,  # User can add custom agent
        'description': 'Reserved for additional features',
        'estimated_time': 'N/A'
    }
]


def launch_agent(agent_config):
    """Launch a single agent."""
    agent_id = agent_config['id']
    script = agent_config['script']
    
    if script is None:
        print(f"Agent {agent_id}: No script defined, skipping")
        return None
    
    log_file = f"research/agents/agent{agent_id}_launch.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [sys.executable, script]
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent
        )
    
    return process


def monitor_agents(processes):
    """Monitor running agents."""
    print("\n" + "="*70)
    print("MONITORING AGENTS")
    print("="*70)
    
    active = True
    while active:
        active = False
        status_lines = []
        
        for agent, process in zip(AGENTS[:len(processes)], processes):
            if process is None:
                status = "SKIPPED"
            elif process.poll() is None:
                status = "RUNNING"
                active = True
            else:
                status = f"EXITED ({process.returncode})"
            
            status_lines.append(f"  Agent {agent['id']}: {agent['name']:<25} [{status}]")
        
        # Clear screen and print status
        print("\033[H\033[J", end="")  # Clear screen
        print("="*70)
        print("PARALLEL AGENTS STATUS")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        for line in status_lines:
            print(line)
        print()
        print("Press Ctrl+C to stop monitoring (agents will continue)")
        
        if active:
            time.sleep(2)
    
    print("\n" + "="*70)
    print("ALL AGENTS COMPLETED")
    print("="*70)


def print_summary():
    """Print summary of agent outputs."""
    print("\n" + "="*70)
    print("AGENT OUTPUTS SUMMARY")
    print("="*70)
    
    for agent in AGENTS:
        agent_id = agent['id']
        results_file = Path(f"research/agents/agent{agent_id}_results/results.json")
        
        print(f"\nAgent {agent_id}: {agent['name']}")
        print(f"  Task: {agent['description']}")
        
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            print(f"  Status: {results.get('status', 'unknown').upper()}")
            
            # Print specific results
            if agent_id == 1:
                print(f"  Test prediction: {results.get('test_prediction', 'N/A')}")
            elif agent_id == 2:
                spearman = results.get('results', {}).get('spearman')
                if spearman:
                    print(f"  Spearman: {spearman:.4f}")
            elif agent_id == 3:
                components = results.get('components_generated', [])
                print(f"  Components: {', '.join(components)}")
        else:
            print(f"  Status: No results file")
        
        # Check for output files
        output_dir = Path(f"research/agents/agent{agent_id}_results")
        if output_dir.exists():
            files = list(output_dir.glob('*'))
            print(f"  Output files: {len(files)}")


def main():
    """Main launcher."""
    print("="*70)
    print("PARALLEL AGENTS LAUNCHER")
    print("="*70)
    print()
    print("Launching 4 agents simultaneously:")
    print()
    
    for agent in AGENTS:
        print(f"  Agent {agent['id']}: {agent['name']}")
        print(f"    └─ {agent['description']} (ETA: {agent['estimated_time']})")
    
    print()
    input("Press Enter to launch agents...")
    
    # Launch agents
    print("\nLaunching agents...")
    processes = []
    
    for agent in AGENTS:
        print(f"  Starting Agent {agent['id']}...")
        process = launch_agent(agent)
        processes.append(process)
        time.sleep(1)  # Stagger launches
    
    print("\n✅ All agents launched!")
    
    # Monitor
    try:
        monitor_agents(processes)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Agents continue running in background.")
    
    # Summary
    print_summary()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review agent outputs in research/agents/")
    print("2. Integrate code into streamlit_app.py")
    print("3. Test dashboard with new features")
    print("4. Deploy updated version")


if __name__ == '__main__':
    main()
