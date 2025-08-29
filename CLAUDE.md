# Claude Code Context

## Project Structure
- `data/` - Dataset loading and management scripts
- `evals/` - Benchmark evaluation scripts
- `.venv/` - Python virtual environment

## Key Commands
- `python evals/benchmark_eval.py` - Run dynamic benchmark evaluations
- `python data/load_data.py` - Download HuggingFace datasets
- `python data/push_data_braintrust.py` - Push datasets to Braintrust

## Environment Variables
- `BRAINTRUST_API_KEY` - Required for Braintrust API access
- `EVAL_MODELS` - Comma-separated list of models to evaluate
- `EVAL_DATASETS` - Comma-separated list of datasets to use

## Notes
- Use `source .venv/bin/activate` to activate virtual environment
- Evaluations create experiments in Braintrust with URLs for results
- Factuality scorer is used for all evaluations currently