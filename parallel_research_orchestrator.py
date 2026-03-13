#!/usr/bin/env python3
"""
Parallel Research Orchestrator

Manages 4 parallel agents:
1. Agent 1: EXP-031 Production Testing - Deploy and test on real FPL
2. Agent 2: 2024-25 Data Collection - Fetch current season data
3. Agent 3: EXP-032 Research Loop - Find better models (Ralph Loop)
4. Agent 4: EXP-032 Research Loop - Find better models (Ralph Loop)

Goal: Continuous improvement while maintaining production reliability
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import threading
import queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ParallelOrchestrator')


@dataclass
class AgentTask:
    """Defines an agent task."""
    id: int
    name: str
    script: str
    args: List[str]
    log_file: str
    status: str = 'pending'
    pid: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result: Optional[Dict] = None


class ParallelResearchOrchestrator:
    """Orchestrates parallel research agents."""
    
    def __init__(self, max_agents=4):
        self.max_agents = max_agents
        self.agents: List[AgentTask] = []
        self.results_dir = Path('research/agents')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Research goals
        self.goals = {
            'exp031_production': {
                'target': 'Deploy EXP-031 and confirm ranking improvement',
                'success_criteria': '73% Spearman maintained in production',
                'deliverable': 'Production comparison report'
            },
            'data_collection': {
                'target': 'Collect 2024-25 season data',
                'success_criteria': '15,000+ new samples added',
                'deliverable': 'Updated dataset with current season'
            },
            'exp032_research': {
                'target': 'Find model beating EXP-031',
                'success_criteria': 'Spearman > 0.75 or RMSE < 1.40',
                'deliverable': 'New champion model'
            }
        }
        
        logger.info("Parallel Research Orchestrator initialized")
        logger.info(f"Research goals: {json.dumps(self.goals, indent=2)}")
    
    def create_agent_tasks(self):
        """Create the 4 parallel agent tasks."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.agents = [
            AgentTask(
                id=1,
                name='EXP-031 Production Testing',
                script='agents/test_exp031_production.py',
                args=['--team-id', '9777842', '--gws', '4'],
                log_file=f'research/agents/agent1_production_{timestamp}.log',
                status='pending'
            ),
            AgentTask(
                id=2,
                name='2024-25 Data Collection',
                script='agents/collect_2024_25_data.py',
                args=['--season', '2024-25'],
                log_file=f'research/agents/agent2_data_{timestamp}.log',
                status='pending'
            ),
            AgentTask(
                id=3,
                name='EXP-032 Research Loop (Alpha)',
                script='agents/ralph_loop_agent.py',
                args=['--agent-id', '3', '--strategy', 'feature_expansion', '--max-iter', '20'],
                log_file=f'research/agents/agent3_ralph_{timestamp}.log',
                status='pending'
            ),
            AgentTask(
                id=4,
                name='EXP-032 Research Loop (Beta)',
                script='agents/ralph_loop_agent.py',
                args=['--agent-id', '4', '--strategy', 'deep_learning', '--max-iter', '20'],
                log_file=f'research/agents/agent4_ralph_{timestamp}.log',
                status='pending'
            )
        ]
        
        logger.info(f"Created {len(self.agents)} agent tasks")
        return self.agents
    
    def launch_agent(self, task: AgentTask) -> subprocess.Popen:
        """Launch a single agent."""
        cmd = [
            sys.executable,
            task.script
        ] + task.args
        
        log_path = self.results_dir / task.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd='/home/akshit/fpl-lineup-optimizer'
            )
        
        task.pid = process.pid
        task.status = 'running'
        task.start_time = datetime.now().isoformat()
        
        logger.info(f"Launched Agent {task.id} ({task.name}) with PID {process.pid}")
        return process
    
    def monitor_agents(self, check_interval=30):
        """Monitor running agents and collect results."""
        processes: Dict[int, subprocess.Popen] = {}
        
        # Launch all agents
        for task in self.agents:
            processes[task.id] = self.launch_agent(task)
            time.sleep(2)  # Stagger launches
        
        logger.info("All agents launched, beginning monitoring...")
        
        # Monitor loop
        active = True
        while active:
            active = False
            status_lines = []
            
            for task in self.agents:
                if task.status == 'running':
                    process = processes[task.id]
                    retcode = process.poll()
                    
                    if retcode is not None:
                        # Process finished
                        task.status = 'completed' if retcode == 0 else 'failed'
                        task.end_time = datetime.now().isoformat()
                        task.result = {
                            'return_code': retcode,
                            'log_file': task.log_file
                        }
                        logger.info(f"Agent {task.id} finished with status: {task.status}")
                    else:
                        active = True
                        
                status_lines.append(
                    f"  Agent {task.id}: {task.name:<30} [{task.status.upper()}]"
                )
            
            # Print status
            os.system('clear' if os.name != 'nt' else 'cls')
            print("="*80)
            print("PARALLEL RESEARCH ORCHESTRATOR")
            print("="*80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("AGENT STATUS:")
            for line in status_lines:
                print(line)
            print()
            print("RESEARCH GOALS:")
            for goal_name, goal in self.goals.items():
                print(f"  • {goal_name}: {goal['target']}")
            print()
            print(f"Checking again in {check_interval}s... (Ctrl+C to stop)")
            
            if active:
                time.sleep(check_interval)
        
        logger.info("All agents completed")
        return self.generate_report()
    
    def generate_report(self):
        """Generate final report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'orchestrator': 'ParallelResearchOrchestrator',
            'agents': [asdict(task) for task in self.agents],
            'summary': {
                'total': len(self.agents),
                'completed': sum(1 for a in self.agents if a.status == 'completed'),
                'failed': sum(1 for a in self.agents if a.status == 'failed'),
                'pending': sum(1 for a in self.agents if a.status == 'pending')
            }
        }
        
        report_path = self.results_dir / 'orchestrator_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run(self):
        """Run the complete orchestration."""
        print("="*80)
        print("PARALLEL RESEARCH ORCHESTRATOR")
        print("="*80)
        print()
        print("This will launch 4 parallel agents:")
        print()
        print("  Agent 1: EXP-031 Production Testing")
        print("           → Deploy and test on real FPL data")
        print("           → Confirm 73% Spearman in production")
        print()
        print("  Agent 2: 2024-25 Data Collection")
        print("           → Fetch current season data")
        print("           → Target: 15,000+ new samples")
        print()
        print("  Agent 3: EXP-032 Research (Feature Expansion)")
        print("           → Try new features to beat EXP-031")
        print("           → Target: Spearman > 0.75")
        print()
        print("  Agent 4: EXP-032 Research (Deep Learning)")
        print("           → Try LSTM/Neural networks")
        print("           → Target: RMSE < 1.40")
        print()
        print("="*80)
        
        # Create tasks
        self.create_agent_tasks()
        
        # Start monitoring
        try:
            report = self.monitor_agents()
            
            print("\n" + "="*80)
            print("ORCHESTRATION COMPLETE")
            print("="*80)
            print(f"Completed: {report['summary']['completed']}/{report['summary']['total']}")
            print(f"Failed: {report['summary']['failed']}")
            print()
            print("Results saved to: research/agents/")
            print()
            
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user")
            print("\n\nStopping agents...")
            for task in self.agents:
                if task.status == 'running' and task.pid:
                    try:
                        os.kill(task.pid, 9)
                        logger.info(f"Killed Agent {task.id} (PID {task.pid})")
                    except:
                        pass


def main():
    orchestrator = ParallelResearchOrchestrator(max_agents=4)
    orchestrator.run()


if __name__ == '__main__':
    main()
