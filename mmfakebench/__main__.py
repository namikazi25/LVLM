#!/usr/bin/env python3
"""
MisinfoBench: Multimodal Misinformation Detection Benchmarking Toolkit

Command-line interface for running multimodal misinformation detection benchmarks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('misinfobench.log')
    ]
)

logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core.runner import BenchmarkRunner
    from core.config import ConfigManager
    from core.io import CSVExporter, ResultsWriter, Logger, ConfigLoader, FileManager
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


def parse_overrides(override_strings: list) -> Dict[str, Any]:
    """Parse command line override strings into a nested dictionary.
    
    Args:
        override_strings: List of strings in format 'key.subkey=value'
        
    Returns:
        Nested dictionary with override values
    """
    overrides = {}
    
    for override_str in override_strings:
        if '=' not in override_str:
            logger.warning(f"Invalid override format: {override_str}. Expected 'key=value'")
            continue
            
        key_path, value = override_str.split('=', 1)
        keys = key_path.split('.')
        
        # Try to parse value as JSON, fallback to string
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        # Build nested dictionary
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = parsed_value
    
    return overrides

def main():
    """Main entry point for MisinfoBench CLI."""
    parser = argparse.ArgumentParser(
        description="MisinfoBench: Multimodal Misinformation Detection Benchmarking Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --config configs/misinfobench_baseline.yml --output results/
  %(prog)s preview --dataset misinfobench --samples 10
  %(prog)s validate --config configs/test.yml
  %(prog)s run --config test.yml --override model.name=gpt-4o dataset.batch_size=32
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run benchmark evaluation')
    run_parser.add_argument('--config', '-c', required=True,
                           help='Path to configuration file')
    run_parser.add_argument('--output', '-o',
                           help='Output directory (overrides config)')
    run_parser.add_argument('--model', '-m',
                           help='Model name (overrides config)')
    run_parser.add_argument('--dataset', '-d',
                           help='Dataset path (overrides config)')
    run_parser.add_argument('--override', action='append', default=[],
                           help='Override config values (e.g., model.temperature=0.5)')
    run_parser.add_argument('--dry-run', action='store_true',
                           help='Validate configuration without running')
    run_parser.add_argument('--resume', 
                           help='Resume from checkpoint file')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview dataset samples')
    preview_parser.add_argument('--dataset', '-d', required=True,
                               help='Dataset configuration file or type')
    preview_parser.add_argument('--samples', '-n', type=int, default=5,
                               help='Number of samples to preview')
    preview_parser.add_argument('--output', '-o',
                               help='Save preview to file')
    preview_parser.add_argument('--format', choices=['json', 'yaml', 'table'],
                               default='table', help='Preview output format')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config', '-c', required=True,
                                help='Path to configuration file')
    validate_parser.add_argument('--strict', action='store_true',
                                help='Enable strict validation')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available components')
    list_parser.add_argument('component', choices=['models', 'datasets', 'modules'],
                            help='Component type to list')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--config', '-c',
                            help='Show configuration details')
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            handle_run_command(args)
        elif args.command == 'preview':
            handle_preview_command(args)
        elif args.command == 'validate':
            handle_validate_command(args)
        elif args.command == 'list':
            handle_list_command(args)
        elif args.command == 'info':
            handle_info_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def handle_run_command(args):
    """Handle the run command."""
    logger.info(f"Loading configuration from: {args.config}")
    
    # Load and validate configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Apply command line overrides
    if args.output:
        config['output']['directory'] = args.output
    if args.model:
        config['model']['name'] = args.model
    if args.dataset:
        config['dataset']['data_path'] = args.dataset
    
    # Apply custom overrides
    if args.override:
        overrides = parse_overrides(args.override)
        config = config_manager.merge_configs(config, overrides)
    
    # Validate final configuration
    validation = config_manager.validator.validate_benchmark_config(config)
    if not validation['valid']:
        raise ValueError(f"Invalid configuration: {validation['errors']}")
    
    if args.dry_run:
        logger.info("Configuration validation successful")
        logger.info(f"Final configuration: {json.dumps(config, indent=2)}")
        return
    
    # Initialize and run benchmark
    runner = BenchmarkRunner(config)
    
    # Setup components
    runner.setup_model_router(config['model'])
    runner.setup_dataset(config['dataset'])
    runner.setup_pipeline(config['pipeline'])
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        results = runner.resume(args.resume)
    else:
        logger.info("Starting benchmark run")
        limit = config['dataset'].get('params', {}).get('limit')
        results = runner.run_benchmark(limit=limit)
    
    logger.info(f"Benchmark completed. Results saved to: {results.get('output_path')}")


def handle_preview_command(args):
    """Handle the preview command."""
    logger.info(f"Previewing dataset: {args.dataset}")
    
    # Determine if dataset argument is a file path or dataset type
    if Path(args.dataset).exists():
        # Load dataset from configuration file
        config_manager = ConfigManager()
        config = config_manager.load_config(args.dataset)
        dataset_config = config['dataset']
    else:
        # Assume it's a dataset type
        if args.dataset in ['misinfobench', 'mocheg']:
            dataset_config = {
                'type': args.dataset,
                'data_path': f'data/{args.dataset}',
                'params': {'limit': args.samples}
            }
        else:
            logger.error(f"Unknown dataset type: {args.dataset}")
            return
    
    # Load dataset
    from datasets.pipeline import create_dataset
    dataset = create_dataset(dataset_config)
    
    # Get samples
    samples = dataset.load()[:args.samples]
    
    # Format output
    if args.format == 'json':
        output = json.dumps(samples, indent=2, default=str)
    elif args.format == 'yaml':
        import yaml
        output = yaml.dump(samples, default_flow_style=False)
    else:  # table format
        output = format_samples_table(samples)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info(f"Preview saved to: {args.output}")
    else:
        print(output)


def handle_validate_command(args):
    """Handle the validate command."""
    logger.info(f"Validating configuration: {args.config}")
    
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config(args.config)
        config_manager.validate_config(config, strict=args.strict)
        logger.info("✓ Configuration is valid")
        
        # Show configuration summary
        print("\nConfiguration Summary:")
        print(f"  Model: {config['model']['name']}")
        print(f"  Dataset: {config['dataset']['type']} ({config['dataset']['data_path']})")
        print(f"  Pipeline: {len(config['pipeline'])} modules")
        print(f"  Output: {config['output']['directory']}")
        
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        raise


def handle_list_command(args):
    """Handle the list command."""
    if args.component == 'models':
        print("Available Models:")
        models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            "gemini-pro", "gemini-pro-vision"
        ]
        for model in models:
            print(f"  - {model}")
    
    elif args.component == 'datasets':
        print("Available Datasets:")
        datasets = ["misinfobench", "mocheg"]
        for dataset in datasets:
            print(f"  - {dataset}")
    
    elif args.component == 'modules':
        print("Available Pipeline Modules:")
        modules = [
            "relevance_checker", "claim_enrichment", "question_generator",
            "web_searcher", "evidence_tagger", "synthesizer"
        ]
        for module in modules:
            print(f"  - {module}")


def handle_info_command(args):
    """Handle the info command."""
    print("MisinfoBench System Information")
    print("=" * 40)
    
    # System info
    import platform
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    # Package info
    try:
        import pkg_resources
        print(f"Project Root: {project_root}")
    except ImportError:
        pass
    
    # Configuration info
    if args.config:
        print(f"\nConfiguration: {args.config}")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        print(json.dumps(config, indent=2))


def format_samples_table(samples):
    """Format samples as a readable table."""
    if not samples:
        return "No samples found."
    
    output = []
    output.append("Dataset Preview")
    output.append("=" * 50)
    
    for i, sample in enumerate(samples, 1):
        output.append(f"\nSample {i}:")
        output.append(f"  ID: {sample.get('id', 'N/A')}")
        output.append(f"  Text: {sample.get('text', 'N/A')[:100]}...")
        output.append(f"  Image: {sample.get('image_path', 'N/A')}")
        output.append(f"  Label: {sample.get('label', 'N/A')}")
        if 'metadata' in sample:
            output.append(f"  Metadata: {sample['metadata']}")
    
    return "\n".join(output)


if __name__ == '__main__':
    sys.exit(main())