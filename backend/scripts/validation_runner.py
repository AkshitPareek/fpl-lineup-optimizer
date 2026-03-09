#!/usr/bin/env python3
"""
Comprehensive Validation Runner for FPL ML Pipeline

Runs all validation tests and generates a comprehensive report.
This should be run before deploying models to production or after
making significant changes.

Usage:
    # Run all validations
    python validation_runner.py --all
    
    # Run specific validation suites
    python validation_runner.py --data --models
    
    # Run with verbose output
    python validation_runner.py --all --verbose
    
    # Generate HTML report
    python validation_runner.py --all --report validation_report.html
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time


class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class ValidationRunner:
    """Runs validation tests and generates reports."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0,
            }
        }
    
    def run_command(self, cmd: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        if self.verbose:
            print_info(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            elapsed = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed_seconds': elapsed,
                'description': description,
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout',
                'elapsed_seconds': time.time() - start_time,
                'description': description,
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'elapsed_seconds': time.time() - start_time,
                'description': description,
            }
    
    def validate_data(self) -> Dict[str, Any]:
        """Run data validation tests."""
        print_header("DATA VALIDATION")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/test_data_validation.py',
            '-v' if self.verbose else '-q',
            '--tb=short',
        ]
        
        result = self.run_command(cmd, "Data Validation")
        
        # Parse results
        if result['success']:
            print_success("All data validation tests passed")
        else:
            print_error("Some data validation tests failed")
            if self.verbose and result.get('stdout'):
                print(result['stdout'])
        
        self.results['tests']['data_validation'] = result
        return result
    
    def validate_models(self) -> Dict[str, Any]:
        """Run model validation tests."""
        print_header("MODEL VALIDATION")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/test_model_validation.py',
            '-v' if self.verbose else '-q',
            '--tb=short',
        ]
        
        result = self.run_command(cmd, "Model Validation")
        
        if result['success']:
            print_success("All model validation tests passed")
        else:
            print_error("Some model validation tests failed")
            if self.verbose and result.get('stdout'):
                print(result['stdout'])
        
        self.results['tests']['model_validation'] = result
        return result
    
    def validate_regression(self) -> Dict[str, Any]:
        """Run regression tests."""
        print_header("REGRESSION TESTS")
        
        # Check if baseline exists
        baseline_path = Path('benchmark_results/baseline_metrics.json')
        if not baseline_path.exists():
            print_warning("No baseline metrics found. Run 'establish-baseline' first.")
            print_info("Skipping regression tests")
            return {
                'success': True,  # Not a failure, just skipped
                'skipped': True,
                'reason': 'No baseline metrics',
            }
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/test_model_regression.py',
            '-v' if self.verbose else '-q',
            '--tb=short',
        ]
        
        result = self.run_command(cmd, "Regression Tests")
        
        if result['success']:
            print_success("All regression tests passed")
        else:
            print_error("Some regression tests failed - performance may have regressed!")
            if self.verbose and result.get('stdout'):
                print(result['stdout'])
        
        self.results['tests']['regression_tests'] = result
        return result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests."""
        print_header("UNIT TESTS")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/',
            '-v' if self.verbose else '-q',
            '--tb=short',
            '-x',  # Stop on first failure
        ]
        
        result = self.run_command(cmd, "Unit Tests")
        
        if result['success']:
            print_success("All unit tests passed")
        else:
            print_error("Some unit tests failed")
            if self.verbose and result.get('stdout'):
                print(result['stdout'])
        
        self.results['tests']['unit_tests'] = result
        return result
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run model benchmarks."""
        print_header("MODEL BENCHMARK")
        
        # Check which models are available
        available_models = []
        for model in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'ridge', 'ensemble']:
            path = Path(f'models/{model}/model.pkl')
            if path.exists():
                available_models.append(model)
        
        if len(available_models) < 2:
            print_warning("Need at least 2 models for benchmark")
            return {
                'success': True,
                'skipped': True,
                'reason': 'Insufficient models',
            }
        
        cmd = [
            sys.executable,
            'backend/scripts/ab_testing_cli.py',
            'benchmark',
            '--models'] + available_models + [
            '--report',
        ]
        
        result = self.run_command(cmd, "Model Benchmark")
        
        if result['success']:
            print_success("Benchmark completed successfully")
            # Find and report the report path
            if 'benchmark_results' in result.get('stdout', ''):
                print_info("Check benchmark_results/ for detailed report")
        else:
            print_error("Benchmark failed")
            if self.verbose and result.get('stdout'):
                print(result['stdout'])
        
        self.results['tests']['benchmark'] = result
        return result
    
    def run_quick_smoke_test(self) -> Dict[str, Any]:
        """Run a quick smoke test to check basic functionality."""
        print_header("SMOKE TEST")
        
        # Test that we can load models and make predictions
        try:
            import numpy as np
            import joblib
            
            # Load test data
            X_test = np.load('datasets/fpl_points_v1/test_X.npy')
            
            # Try to load and predict with each model
            models_tested = []
            for model_name in ['xgboost', 'lightgbm', 'ridge']:
                path = f'models/{model_name}/model.pkl'
                if os.path.exists(path):
                    model = joblib.load(path)
                    preds = model.predict(X_test[:5])
                    assert len(preds) == 5
                    models_tested.append(model_name)
            
            print_success(f"Smoke test passed ({len(models_tested)} models tested)")
            return {
                'success': True,
                'models_tested': models_tested,
            }
        
        except Exception as e:
            print_error(f"Smoke test failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all test results."""
        total = len(self.results['tests'])
        passed = sum(1 for r in self.results['tests'].values() if r.get('success'))
        failed = total - passed
        
        self.results['summary'] = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
        }
        
        return self.results['summary']
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.generate_summary()
        
        print_header("VALIDATION SUMMARY")
        
        for test_name, result in self.results['tests'].items():
            status = "✓ PASS" if result.get('success') else "✗ FAIL"
            color = Colors.GREEN if result.get('success') else Colors.RED
            
            if result.get('skipped'):
                status = "⊘ SKIP"
                color = Colors.YELLOW
            
            elapsed = result.get('elapsed_seconds', 0)
            print(f"{color}{status}{Colors.END} {test_name:<25} ({elapsed:.1f}s)")
        
        print(f"\n{Colors.BOLD}Total: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.0%}){Colors.END}")
        
        if summary['failed'] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All validations passed! Ready for deployment.{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ Some validations failed. Please review errors above.{Colors.END}")
    
    def save_report(self, output_path: str):
        """Save validation report to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print_info(f"Report saved to {output_path}")
    
    def generate_html_report(self, output_path: str):
        """Generate HTML report."""
        summary = self.results['summary']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FPL ML Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: {'#d4edda' if summary['failed'] == 0 else '#f8d7da'}; 
                    padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .test-result {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .pass {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .fail {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .skip {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .metric {{ font-family: monospace; font-size: 1.1em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 FPL ML Validation Report</h1>
        <p class="timestamp">Generated: {self.results['timestamp']}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Tests:</strong> {summary['total']}</p>
            <p><strong>Passed:</strong> <span style="color: green;">{summary['passed']}</span></p>
            <p><strong>Failed:</strong> <span style="color: {'green' if summary['failed'] == 0 else 'red'};">{summary['failed']}</span></p>
            <p><strong>Pass Rate:</strong> {summary['pass_rate']:.1%}</p>
            <p><strong>Status:</strong> {'✅ Ready for Deployment' if summary['failed'] == 0 else '❌ Issues Found'}</p>
        </div>
        
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Time (s)</th>
                <th>Details</th>
            </tr>
"""
        
        for test_name, result in self.results['tests'].items():
            status_class = 'pass' if result.get('success') else 'fail'
            status_text = 'PASS' if result.get('success') else 'FAIL'
            
            if result.get('skipped'):
                status_class = 'skip'
                status_text = 'SKIP'
            
            elapsed = result.get('elapsed_seconds', 0)
            details = result.get('error', '') if not result.get('success') else ''
            
            html += f"""
            <tr class="{status_class}">
                <td>{test_name}</td>
                <td><strong>{status_text}</strong></td>
                <td>{elapsed:.1f}s</td>
                <td>{details}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Next Steps</h2>
"""
        
        if summary['failed'] == 0:
            html += """
        <ul>
            <li>✅ All validations passed</li>
            <li>Models are ready for production deployment</li>
            <li>Consider establishing new baseline metrics</li>
        </ul>
"""
        else:
            html += """
        <ul>
            <li>❌ Review failed tests above</li>
            <li>Fix any data quality issues</li>
            <li>Retrain models if necessary</li>
            <li>Re-run validation before deployment</li>
        </ul>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print_info(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Validation Runner for FPL ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validations
  python validation_runner.py --all
  
  # Quick smoke test only
  python validation_runner.py --smoke
  
  # Data and model validation only
  python validation_runner.py --data --models
  
  # Full validation with HTML report
  python validation_runner.py --all --html-report report.html
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all validations')
    parser.add_argument('--data', action='store_true', help='Run data validation')
    parser.add_argument('--models', action='store_true', help='Run model validation')
    parser.add_argument('--regression', action='store_true', help='Run regression tests')
    parser.add_argument('--unit-tests', action='store_true', help='Run unit tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', help='Save JSON report to file')
    parser.add_argument('--html-report', help='Save HTML report to file')
    
    args = parser.parse_args()
    
    # If no specific tests selected, default to smoke test
    if not any([args.all, args.data, args.models, args.regression, 
                args.unit_tests, args.benchmark, args.smoke]):
        args.smoke = True
    
    # --all enables everything
    if args.all:
        args.data = True
        args.models = True
        args.regression = True
        args.unit_tests = True
        args.benchmark = True
        args.smoke = True
    
    runner = ValidationRunner(verbose=args.verbose)
    
    print_header("FPL ML VALIDATION RUNNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run selected validations
    if args.smoke:
        runner.run_quick_smoke_test()
    
    if args.data:
        runner.validate_data()
    
    if args.models:
        runner.validate_models()
    
    if args.regression:
        runner.validate_regression()
    
    if args.unit_tests:
        runner.run_unit_tests()
    
    if args.benchmark:
        runner.run_benchmark()
    
    # Print summary
    runner.print_summary()
    
    # Save reports
    if args.report:
        runner.save_report(args.report)
    
    if args.html_report:
        runner.generate_html_report(args.html_report)
    
    # Exit with appropriate code
    summary = runner.results['summary']
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
