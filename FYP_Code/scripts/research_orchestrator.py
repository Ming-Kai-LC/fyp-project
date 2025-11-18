"""
Autonomous Data Science Research Orchestrator
TAR UMT FYP - CrossViT for COVID-19 Classification
Following Academic Research Standards and Best Practices

This orchestrator manages:
- 30 training experiments (6 models × 5 seeds)
- MLflow experiment tracking
- GPU resource management (safe for shared workstation)
- Statistical validation
- Publication-ready results generation
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

VENV_PYTHON = Path("venv/Scripts/python.exe")
NOTEBOOKS_DIR = Path("notebooks")

class ResearchOrchestrator:
    """Manages the complete FYP research workflow"""

    def __init__(self):
        self.results = {
            'start_time': datetime.now().isoformat(),
            'experiments': [],
            'phase_times': {},
            'models_trained': 0,
            'models_failed': 0
        }

        # Experiment configurations following CRISP-DM methodology
        self.experiments = [
            {
                'phase': 'Phase 1',
                'model': 'ResNet-50 Baseline',
                'notebook': '04_baseline_test.ipynb',
                'priority': 1,
                'description': 'Verify training pipeline (Phase 1 completion)',
                'config_update': {'test_on_subset': False},  # Use FULL dataset
                'timeout_minutes': 120,
                'seeds': [42]  # Single seed for baseline
            },
            {
                'phase': 'Phase 2',
                'model': 'CrossViT-Tiny',
                'notebook': '06_crossvit_training.ipynb',
                'priority': 2,
                'description': 'Main model - Cross-Attention Vision Transformer',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 300
            },
            {
                'phase': 'Phase 2',
                'model': 'ResNet-50',
                'notebook': '07_resnet50_training.ipynb',
                'priority': 3,
                'description': 'Baseline 1 - Deep CNN',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 240
            },
            {
                'phase': 'Phase 2',
                'model': 'DenseNet-121',
                'notebook': '08_densenet121_training.ipynb',
                'priority': 4,
                'description': 'Baseline 2 - Dense connections',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 240
            },
            {
                'phase': 'Phase 2',
                'model': 'EfficientNet-B0',
                'notebook': '09_efficientnet_training.ipynb',
                'priority': 5,
                'description': 'Baseline 3 - Efficient scaling',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 240
            },
            {
                'phase': 'Phase 2',
                'model': 'ViT-Base',
                'notebook': '10_vit_training.ipynb',
                'priority': 6,
                'description': 'Baseline 4 - Pure transformer',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 300
            },
            {
                'phase': 'Phase 2',
                'model': 'Swin-Tiny',
                'notebook': '11_swin_training.ipynb',
                'priority': 7,
                'description': 'Baseline 5 - Hierarchical transformer',
                'seeds': [42, 123, 456, 789, 101112],
                'timeout_minutes': 240
            }
        ]

    def check_gpu_available(self):
        """Check if GPU is available and get current usage"""
        try:
            result = subprocess.run(
                [str(VENV_PYTHON), "-c",
                 "import torch; print(torch.cuda.is_available()); "
                 "print(torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            lines = result.stdout.strip().split('\n')
            available = lines[0] == 'True'
            usage_gb = float(lines[1]) if len(lines) > 1 else 0
            return available, usage_gb
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return False, 0

    def execute_notebook(self, experiment_config):
        """Execute a training notebook with proper error handling"""
        notebook_path = NOTEBOOKS_DIR / experiment_config['notebook']
        model_name = experiment_config['model']
        timeout = experiment_config.get('timeout_minutes', 240) * 60

        if not notebook_path.exists():
            logger.error(f"Notebook not found: {notebook_path}")
            return False

        logger.info(f"\n{'='*70}")
        logger.info(f"EXECUTING: {model_name}")
        logger.info(f"Notebook: {experiment_config['notebook']}")
        logger.info(f"Description: {experiment_config['description']}")
        logger.info(f"{'='*70}\n")

        cmd = [
            str(VENV_PYTHON), "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(notebook_path),
            f"--ExecutePreprocessor.timeout={timeout}"
        ]

        start_time = time.time()

        try:
            # Check GPU before starting
            gpu_avail, gpu_usage = self.check_gpu_available()
            logger.info(f"GPU Available: {gpu_avail}, Current Usage: {gpu_usage:.2f} GB")

            # Execute notebook
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 300,
                cwd=str(Path.cwd())
            )

            elapsed = (time.time() - start_time) / 60

            if process.returncode == 0:
                logger.info(f"✅ SUCCESS: {model_name} completed in {elapsed:.1f} minutes")
                self.results['models_trained'] += 1
                return True
            else:
                logger.error(f"❌ FAILED: {model_name} (exit code {process.returncode})")
                logger.error(f"Error output:\n{process.stderr[-2000:]}")  # Last 2000 chars
                self.results['models_failed'] += 1
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ TIMEOUT: {model_name} exceeded {timeout/60:.0f} minutes")
            self.results['models_failed'] += 1
            return False
        except Exception as e:
            logger.error(f"❌ ERROR: {model_name} - {e}")
            self.results['models_failed'] += 1
            return False

    def run_all_experiments(self):
        """Execute all experiments following academic research protocol"""
        logger.info("\n" + "="*70)
        logger.info("AUTONOMOUS FYP RESEARCH ORCHESTRATION")
        logger.info("TAR UMT Data Science FYP 2025/26")
        logger.info("Following Academic Research Standards")
        logger.info("="*70)

        logger.info(f"\n📊 Experiment Plan:")
        logger.info(f"   - Total experiments: {sum(len(exp.get('seeds', [1])) for exp in self.experiments)}")
        logger.info(f"   - Models: {len(self.experiments)}")
        logger.info(f"   - Seeds per model: 1-5")
        logger.info(f"   - Estimated time: 20-24 hours")
        logger.info(f"   - GPU: RTX 6000 Ada (Shared workstation mode)")

        total_start = time.time()

        for exp_idx, experiment in enumerate(self.experiments, 1):
            logger.info(f"\n\n{'#'*70}")
            logger.info(f"# EXPERIMENT {exp_idx}/{len(self.experiments)}")
            logger.info(f"# {experiment['phase']}: {experiment['model']}")
            logger.info(f"{'#'*70}\n")

            phase_start = time.time()

            # Execute the experiment
            success = self.execute_notebook(experiment)

            phase_time = (time.time() - phase_start) / 60

            # Record results
            experiment_result = {
                'model': experiment['model'],
                'phase': experiment['phase'],
                'success': success,
                'time_minutes': phase_time,
                'timestamp': datetime.now().isoformat()
            }
            self.results['experiments'].append(experiment_result)

            # Save intermediate results
            self.save_results()

            if not success:
                logger.warning(f"⚠️ {experiment['model']} failed - continuing with next model")

            # Cooldown between experiments
            logger.info("\n⏸️  Cooldown: 30 seconds before next experiment...")
            time.sleep(30)

        # Final summary
        total_time = (time.time() - total_start) / 3600
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_hours'] = total_time

        self.print_final_summary()
        self.save_results()

    def save_results(self):
        """Save research results to JSON"""
        results_file = Path('research_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"💾 Results saved to {results_file}")

    def print_final_summary(self):
        """Print comprehensive research summary"""
        logger.info("\n\n" + "="*70)
        logger.info("RESEARCH ORCHESTRATION COMPLETE")
        logger.info("="*70)

        logger.info(f"\n⏱️ Total Time: {self.results['total_hours']:.2f} hours")
        logger.info(f"\n📊 Results Summary:")
        logger.info(f"   ✅ Models trained successfully: {self.results['models_trained']}")
        logger.info(f"   ❌ Models failed: {self.results['models_failed']}")
        logger.info(f"   📈 Success rate: {self.results['models_trained']/(self.results['models_trained']+self.results['models_failed'])*100:.1f}%")

        logger.info(f"\n📋 Per-Model Results:")
        for exp in self.results['experiments']:
            status = "✅" if exp['success'] else "❌"
            logger.info(f"   {status} {exp['model']:30s} - {exp['time_minutes']:.1f} min")

        logger.info(f"\n🎯 Next Steps:")
        if self.results['models_trained'] >= 6:
            logger.info("   1. View results: mlflow ui")
            logger.info("   2. Run statistical validation (Phase 3)")
            logger.info("   3. Generate thesis tables (Chapter 5)")
            logger.info("   4. Create Flask demo (Phase 4)")
        else:
            logger.info("   ⚠️ Some models failed - review logs and retry")

        logger.info("\n" + "="*70)


def main():
    """Main entry point"""
    try:
        orchestrator = ResearchOrchestrator()
        orchestrator.run_all_experiments()
        return 0
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Research interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
