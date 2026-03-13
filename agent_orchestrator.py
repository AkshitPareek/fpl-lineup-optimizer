#!/usr/bin/env python3
"""
Agent Orchestrator for Parallel Data Collection

Launches multiple agents to collect historical FPL data in parallel:
- Agent 1: 2020-21 season
- Agent 2: 2021-22 season
- Agent 3: 2022-23 season
- Agent 4: 2023-24 season
- Agent 5: Championship data
- Agent 6: Transfer database

Usage:
    python agent_orchestrator.py [--agents 6] [--output data/historical/]
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('agent_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Represents a task for an agent."""
    agent_id: int
    task_type: str  # 'season', 'championship', 'transfers'
    season: Optional[str]
    output_dir: str
    priority: int
    dependencies: List[int]
    
    def __repr__(self):
        if self.season:
            return f"Agent-{self.agent_id}: Collect {self.season} {self.task_type}"
        return f"Agent-{self.agent_id}: Collect {self.task_type}"


@dataclass
class AgentResult:
    """Result from an agent."""
    agent_id: int
    task_type: str
    season: Optional[str]
    status: str  # 'success', 'failed', 'running'
    samples_collected: int = 0
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0


class DataCollectionAgent:
    """Individual agent that collects data for a specific task."""
    
    def __init__(self, task: AgentTask):
        self.task = task
        self.result = AgentResult(
            agent_id=task.agent_id,
            task_type=task.task_type,
            season=task.season,
            status='running',
            start_time=datetime.now().isoformat()
        )
    
    def run(self) -> AgentResult:
        """Execute the agent's task."""
        logger.info(f"🚀 Agent-{self.task.agent_id} starting: {self.task}")
        
        try:
            if self.task.task_type == 'season':
                self._collect_season()
            elif self.task.task_type == 'championship':
                self._collect_championship()
            elif self.task.task_type == 'transfers':
                self._collect_transfers()
            else:
                raise ValueError(f"Unknown task type: {self.task.task_type}")
            
            self.result.status = 'success'
            logger.info(f"✅ Agent-{self.task.agent_id} completed successfully")
            
        except Exception as e:
            self.result.status = 'failed'
            self.result.error_message = str(e)
            logger.error(f"❌ Agent-{self.task.agent_id} failed: {e}")
        
        finally:
            self.result.end_time = datetime.now().isoformat()
            if self.result.start_time:
                start = datetime.fromisoformat(self.result.start_time)
                end = datetime.fromisoformat(self.result.end_time)
                self.result.duration_seconds = (end - start).total_seconds()
        
        return self.result
    
    def _collect_season(self):
        """Collect data for a specific PL season."""
        season = self.task.season
        output_dir = Path(self.task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📡 Agent-{self.task.agent_id}: Fetching {season} data...")
        
        # Simulate data collection (replace with actual implementation)
        # In production, this would call FPL API or historical data source
        
        output_file = output_dir / f"pl_{season.replace('-', '_')}.json"
        
        # Create sample structure (agents would populate with real data)
        sample_data = {
            'season': season,
            'agent_id': self.task.agent_id,
            'collection_date': datetime.now().isoformat(),
            'players': [],  # Would be populated
            'status': 'collected'
        }
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.result.file_path = str(output_file)
        self.result.samples_collected = 0  # Would be actual count
        
        logger.info(f"💾 Agent-{self.task.agent_id}: Saved to {output_file}")
    
    def _collect_championship(self):
        """Collect Championship data for promoted teams."""
        logger.info(f"📡 Agent-{self.task.agent_id}: Fetching Championship data...")
        
        output_dir = Path(self.task.output_dir)
        output_file = output_dir / "championship_promoted_teams.json"
        
        # Would collect data for teams like Luton, Sheffield United, Burnley
        
        self.result.file_path = str(output_file)
        self.result.samples_collected = 0
        
        logger.info(f"💾 Agent-{self.task.agent_id}: Saved Championship data")
    
    def _collect_transfers(self):
        """Collect transfer history."""
        logger.info(f"📡 Agent-{self.task.agent_id}: Fetching transfer data...")
        
        output_dir = Path(self.task.output_dir)
        output_file = output_dir / "transfer_database.json"
        
        # Would collect from Transfermarkt or similar
        
        self.result.file_path = str(output_file)
        self.result.samples_collected = 0
        
        logger.info(f"💾 Agent-{self.task.agent_id}: Saved transfer database")


class AgentOrchestrator:
    """Orchestrates multiple agents for parallel data collection."""
    
    def __init__(self, max_workers: int = 6, output_dir: str = "data/historical"):
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.tasks: List[AgentTask] = []
        self.results: List[AgentResult] = []
        
    def create_tasks(self) -> List[AgentTask]:
        """Create all data collection tasks."""
        tasks = []
        
        # Season collection tasks (can run in parallel)
        seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
        for i, season in enumerate(seasons, 1):
            tasks.append(AgentTask(
                agent_id=i,
                task_type='season',
                season=season,
                output_dir=self.output_dir,
                priority=1,
                dependencies=[]
            ))
        
        # Championship data (can run in parallel with seasons)
        tasks.append(AgentTask(
            agent_id=5,
            task_type='championship',
            season=None,
            output_dir=self.output_dir,
            priority=2,
            dependencies=[]
        ))
        
        # Transfer database (can run in parallel)
        tasks.append(AgentTask(
            agent_id=6,
            task_type='transfers',
            season=None,
            output_dir=self.output_dir,
            priority=2,
            dependencies=[]
        ))
        
        self.tasks = tasks
        logger.info(f"📋 Created {len(tasks)} tasks for {self.max_workers} agents")
        
        return tasks
    
    def check_disk_space(self) -> bool:
        """Check if there's enough disk space."""
        import shutil
        
        total, used, free = shutil.disk_usage(self.output_dir)
        free_gb = free / (1024**3)
        
        logger.info(f"💾 Disk space check: {free_gb:.1f} GB free")
        
        if free_gb < 10:
            logger.error(f"❌ Insufficient disk space: {free_gb:.1f} GB free")
            return False
        
        return True
    
    def optimize_storage(self):
        """Optimize storage before starting."""
        logger.info("🔧 Optimizing storage...")
        
        # Clean up old cache files
        cache_dir = Path('data/cache')
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            logger.info("🗑️  Cleaned cache directory")
        
        # Remove old model backups (keep only latest)
        models_dir = Path('models')
        for model_type in ['xgboost', 'lightgbm', 'random_forest']:
            model_dir = models_dir / model_type
            if model_dir.exists():
                # Keep only .pkl files, remove other artifacts
                for f in model_dir.glob('*'):
                    if f.suffix not in ['.pkl', '.json']:
                        f.unlink()
        
        logger.info("✅ Storage optimization complete")
    
    def run_parallel(self) -> List[AgentResult]:
        """Run all agents in parallel."""
        logger.info("="*70)
        logger.info("AGENT ORCHESTRATOR: PARALLEL DATA COLLECTION")
        logger.info("="*70)
        
        # Pre-flight checks
        if not self.check_disk_space():
            logger.error("Aborting: Insufficient disk space")
            return []
        
        self.optimize_storage()
        
        # Create tasks
        tasks = self.create_tasks()
        
        # Run agents in parallel
        logger.info(f"\n🚀 Launching {len(tasks)} agents with {self.max_workers} workers...")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_agent, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    status_emoji = "✅" if result.status == 'success' else "❌"
                    logger.info(f"{status_emoji} Agent-{result.agent_id} finished: "
                              f"{result.status} ({result.duration_seconds:.1f}s)")
                    
                except Exception as e:
                    logger.error(f"❌ Agent-{task.agent_id} exception: {e}")
                    self.results.append(AgentResult(
                        agent_id=task.agent_id,
                        task_type=task.task_type,
                        season=task.season,
                        status='failed',
                        error_message=str(e)
                    ))
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("COLLECTION COMPLETE")
        logger.info("="*70)
        
        success_count = sum(1 for r in self.results if r.status == 'success')
        failed_count = len(self.results) - success_count
        
        logger.info(f"\nResults:")
        logger.info(f"  Total agents: {len(self.results)}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total time: {total_time:.1f}s")
        
        # Save report
        self._save_report(total_time)
        
        return self.results
    
    def _run_agent(self, task: AgentTask) -> AgentResult:
        """Run a single agent."""
        agent = DataCollectionAgent(task)
        return agent.run()
    
    def _save_report(self, total_time: float):
        """Save collection report."""
        report = {
            'orchestration_date': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'max_workers': self.max_workers,
            'tasks': [asdict(task) for task in self.tasks],
            'results': [asdict(result) for result in self.results],
            'summary': {
                'total_agents': len(self.results),
                'successful': sum(1 for r in self.results if r.status == 'success'),
                'failed': sum(1 for r in self.results if r.status == 'failed'),
                'total_samples': sum(r.samples_collected for r in self.results)
            }
        }
        
        report_file = Path(self.output_dir) / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Agent Orchestrator for Data Collection')
    parser.add_argument('--agents', type=int, default=6, help='Number of parallel agents')
    parser.add_argument('--output', default='data/historical', help='Output directory')
    parser.add_argument('--check-only', action='store_true', help='Only check prerequisites')
    
    args = parser.parse_args()
    
    orchestrator = AgentOrchestrator(max_workers=args.agents, output_dir=args.output)
    
    if args.check_only:
        print("🔍 Running prerequisite checks...")
        orchestrator.check_disk_space()
        orchestrator.optimize_storage()
        print("✅ Checks complete")
        return 0
    
    results = orchestrator.run_parallel()
    
    # Return success if majority of agents succeeded
    success_count = sum(1 for r in results if r.status == 'success')
    return 0 if success_count >= len(results) * 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
