"""
Enhanced CLI interface with rich progress bars, better help, and validation.

This module provides a modern, user-friendly command-line interface using Rich
for beautiful output, progress tracking, and comprehensive error handling.
"""

import click
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import print as rich_print
import psutil

from src.logger import get_logger
from src.config import get_all_configs, DATASET_PATH, RESULTS_DIR
from src.error_handler import validate_system_resources, error_handler
from src.model_cache import model_cache

# Set up logger and console
logger = get_logger(__name__)
console = Console()

class EnhancedCLI:
    """Enhanced CLI with rich interface and progress tracking."""
    
    def __init__(self):
        self.console = console
        self.progress = None
        self.current_tasks = {}
        
    def display_banner(self):
        """Display application banner."""
        banner = """
        ╔══════════════════════════════════════════════╗
        ║        🚗 Car Reviews Analysis with LLMs     ║
        ║              Enhanced Version 2.0            ║
        ╚══════════════════════════════════════════════╝
        """
        panel = Panel(
            banner,
            style="bold blue",
            title="🚀 Welcome",
            subtitle="Advanced NLP Analysis Pipeline"
        )
        self.console.print(panel)
    
    def display_system_info(self):
        """Display system information and configuration."""
        # System resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info_table = Table(title="📊 System Information")
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Status", style="green")
        info_table.add_column("Details", style="yellow")
        
        info_table.add_row(
            "Memory", 
            f"{memory.percent:.1f}% used",
            f"{memory.available // (1024**3):.1f}GB available"
        )
        info_table.add_row(
            "Disk Space",
            f"{disk.percent:.1f}% used", 
            f"{disk.free // (1024**3):.1f}GB free"
        )
        info_table.add_row(
            "Model Cache",
            "Enabled" if model_cache else "Disabled",
            f"{len(model_cache.cache) if model_cache else 0} models cached"
        )
        
        self.console.print(info_table)
    
    def display_config(self):
        """Display current configuration."""
        config = get_all_configs()
        
        config_tree = Tree("⚙️  Configuration")
        
        # Project paths
        paths_branch = config_tree.add("📁 Paths")
        paths_branch.add(f"Dataset: {config['dataset_path']}")
        paths_branch.add(f"Results: {config['results_dir']}")
        
        # Models
        models_branch = config_tree.add("🤖 Models")
        for task, model_config in config['models'].items():
            models_branch.add(f"{task}: {model_config['model_name']}")
        
        # Performance
        perf_branch = config_tree.add("⚡ Performance")
        perf_config = config['performance']
        perf_branch.add(f"Batch Size: {perf_config['batch_size']}")
        perf_branch.add(f"Max Length: {perf_config['max_length']}")
        perf_branch.add(f"Cache Enabled: {perf_config['enable_cache']}")
        
        self.console.print(config_tree)
    
    def create_progress_tracker(self, description: str = "Processing") -> Progress:
        """Create a rich progress tracker."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        self.progress = progress
        return progress
    
    def add_task(self, description: str, total: int = 100) -> TaskID:
        """Add a task to the progress tracker."""
        if self.progress:
            task_id = self.progress.add_task(description, total=total)
            self.current_tasks[description] = task_id
            return task_id
        return None
    
    def update_task(self, description: str, advance: int = 1, **kwargs):
        """Update task progress."""
        if self.progress and description in self.current_tasks:
            task_id = self.current_tasks[description]
            self.progress.update(task_id, advance=advance, **kwargs)
    
    def complete_task(self, description: str):
        """Mark task as completed."""
        if self.progress and description in self.current_tasks:
            task_id = self.current_tasks[description]
            self.progress.update(task_id, completed=True)
            self.console.print(f"✅ {description} completed")
    
    def display_results_summary(self, results: Dict[str, Any]):
        """Display a beautiful summary of analysis results."""
        summary_table = Table(title="📊 Analysis Results Summary")
        summary_table.add_column("Task", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Key Metrics", style="yellow")
        summary_table.add_column("Output", style="blue")
        
        for task_name, task_data in results.items():
            if isinstance(task_data, dict):
                status = "✅ Success" if task_data else "❌ Failed"
                
                # Extract key metrics
                metrics = ""
                if "metrics" in task_data:
                    task_metrics = task_data["metrics"]
                    if isinstance(task_metrics, dict):
                        metrics = f"Accuracy: {task_metrics.get('accuracy', 'N/A'):.3f}"
                elif "results" in task_data and isinstance(task_data["results"], dict):
                    results_data = task_data["results"]
                    if "num_topics" in results_data:
                        metrics = f"Topics: {results_data['num_topics']}"
                
                # Output info
                output = ""
                if "predictions" in task_data:
                    output = f"{len(task_data['predictions'])} predictions"
                elif "results" in task_data:
                    output = "Analysis complete"
                
                summary_table.add_row(
                    task_name.title(),
                    status,
                    metrics or "N/A",
                    output or "N/A"
                )
        
        self.console.print(summary_table)
    
    def display_error_summary(self):
        """Display error summary if any errors occurred."""
        error_summary = error_handler.get_error_summary()
        
        if error_summary["total_errors"] > 0:
            error_panel = Panel(
                f"⚠️  Total Errors: {error_summary['total_errors']}\n"
                f"📋 Error Types: {len(error_summary['error_breakdown'])}\n"
                f"🔧 Fallbacks Available: {len(error_summary['registered_fallbacks'])}",
                title="Error Summary",
                style="yellow"
            )
            self.console.print(error_panel)
            
            # Detailed error breakdown
            if error_summary["error_breakdown"]:
                error_table = Table(title="Error Breakdown")
                error_table.add_column("Error", style="red")
                error_table.add_column("Count", style="yellow")
                
                for error_key, count in error_summary["error_breakdown"].items():
                    error_table.add_row(error_key, str(count))
                
                self.console.print(error_table)
    
    def validate_inputs(self, data_file: str, tasks: List[str]) -> bool:
        """Validate CLI inputs with user-friendly messages."""
        errors = []
        
        # Check data file
        if not Path(data_file).exists():
            errors.append(f"❌ Dataset file not found: {data_file}")
        
        # Check tasks
        valid_tasks = ["sentiment", "translation", "qa", "summarization", "topic", "aspect", "ner", "all"]
        for task in tasks:
            if task not in valid_tasks:
                errors.append(f"❌ Invalid task: {task}. Valid options: {', '.join(valid_tasks)}")
        
        # Check system resources
        try:
            validate_system_resources()
        except Exception as e:
            errors.append(f"⚠️  {str(e)}")
        
        if errors:
            error_panel = Panel(
                "\n".join(errors),
                title="❌ Validation Errors",
                style="red"
            )
            self.console.print(error_panel)
            return False
        
        success_panel = Panel(
            "✅ All inputs validated successfully!",
            title="Validation Complete",
            style="green"
        )
        self.console.print(success_panel)
        return True
    
    def prompt_for_confirmation(self, message: str) -> bool:
        """Prompt user for confirmation with rich styling."""
        return click.confirm(
            click.style(f"🤔 {message}", fg="yellow", bold=True)
        )
    
    def display_help_examples(self):
        """Display helpful usage examples."""
        examples = [
            ("Basic sentiment analysis", "python main.py --task sentiment --visualize"),
            ("Full analysis with results saved", "python main.py --task all --save-results --verbose"),
            ("Topic modeling only", "python main.py --task topic --visualize"),
            ("Custom dataset", "python main.py --data-file /path/to/data.csv --task sentiment"),
            ("Multiple specific tasks", "python main.py --task sentiment,translation,qa --visualize")
        ]
        
        examples_table = Table(title="📖 Usage Examples")
        examples_table.add_column("Description", style="cyan")
        examples_table.add_column("Command", style="green")
        
        for description, command in examples:
            examples_table.add_row(description, command)
        
        self.console.print(examples_table)
    
    def display_model_info(self):
        """Display information about available models."""
        config = get_all_configs()
        
        models_table = Table(title="🤖 Available Models")
        models_table.add_column("Task", style="cyan")
        models_table.add_column("Model", style="green")
        models_table.add_column("Type", style="yellow")
        models_table.add_column("Status", style="blue")
        
        for task, model_config in config['models'].items():
            model_name = model_config['model_name']
            task_type = model_config['task']
            
            # Check if model is cached
            cache_key = model_config['cache_key']
            is_cached = cache_key in model_cache.cache if model_cache else False
            status = "🔄 Cached" if is_cached else "📥 Not Loaded"
            
            models_table.add_row(
                task.title(),
                model_name,
                task_type,
                status
            )
        
        self.console.print(models_table)

# Global CLI instance
enhanced_cli = EnhancedCLI()

@click.command()
@click.option(
    "--data-file",
    default=str(DATASET_PATH),
    help="Path to the car reviews dataset (CSV format)",
    type=click.Path(exists=True),
    show_default=True
)
@click.option(
    "--task",
    default="all",
    help="NLP task(s) to perform. Use comma-separated for multiple tasks.",
    show_default=True
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations and interactive dashboard"
)
@click.option(
    "--save-results",
    is_flag=True,
    help="Save analysis results to disk (JSON, CSV, Excel)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output with detailed progress"
)
@click.option(
    "--show-config",
    is_flag=True,
    help="Display current configuration and system info"
)
@click.option(
    "--show-examples",
    is_flag=True,
    help="Display usage examples and exit"
)
@click.option(
    "--show-models",
    is_flag=True,
    help="Display available models and their status"
)
@click.option(
    "--preload-models",
    is_flag=True,
    help="Preload models into cache for faster execution"
)
@click.option(
    "--max-reviews",
    default=None,
    type=int,
    help="Maximum number of reviews to process (for testing)"
)
def enhanced_main(data_file, task, visualize, save_results, verbose, 
                 show_config, show_examples, show_models, preload_models, max_reviews):
    """
    🚗 Enhanced Car Reviews Analysis with LLMs
    
    Analyze car reviews using state-of-the-art language models with beautiful
    progress tracking, comprehensive error handling, and rich visualizations.
    """
    
    # Display banner
    enhanced_cli.display_banner()
    
    # Handle info commands
    if show_examples:
        enhanced_cli.display_help_examples()
        return
    
    if show_models:
        enhanced_cli.display_model_info()
        return
    
    if show_config:
        enhanced_cli.display_system_info()
        enhanced_cli.display_config()
        return
    
    # Preload models if requested
    if preload_models:
        with enhanced_cli.create_progress_tracker("Preloading models") as progress:
            task_id = progress.add_task("Loading models...", total=100)
            try:
                model_cache.preload_models()
                progress.update(task_id, completed=True)
                enhanced_cli.console.print("✅ Models preloaded successfully!")
            except Exception as e:
                progress.update(task_id, description=f"❌ Failed: {e}")
                enhanced_cli.console.print(f"❌ Failed to preload models: {e}")
        return
    
    # Parse tasks
    tasks = [t.strip() for t in task.split(",")]
    
    # Validate inputs
    if not enhanced_cli.validate_inputs(data_file, tasks):
        raise click.ClickException("Validation failed. Please fix the errors above.")
    
    # Display system info if verbose
    if verbose:
        enhanced_cli.display_system_info()
    
    # Import and run the main analysis
    try:
        # Dynamic import to avoid circular imports
        from main import main as original_main
        import sys
        
        # Prepare arguments for original main
        sys.argv = [
            "main.py",
            "--data-file", data_file,
            "--task", task
        ]
        
        if visualize:
            sys.argv.append("--visualize")
        if save_results:
            sys.argv.append("--save-results")
        if verbose:
            sys.argv.append("--verbose")
        
        # Run with progress tracking
        with enhanced_cli.create_progress_tracker("Analysis Pipeline") as progress:
            # Track overall progress
            main_task = progress.add_task("Running analysis...", total=len(tasks) if tasks != ["all"] else 7)
            
            # Execute analysis
            original_main()
            
            progress.update(main_task, completed=True)
        
        enhanced_cli.console.print("\n🎉 Analysis completed successfully!")
        
        # Display error summary if any errors occurred
        enhanced_cli.display_error_summary()
        
    except KeyboardInterrupt:
        enhanced_cli.console.print("\n⚠️  Analysis interrupted by user")
        raise click.Abort()
    except Exception as e:
        enhanced_cli.console.print(f"\n❌ Analysis failed: {e}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    enhanced_main()