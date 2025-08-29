# HuggingFace Benchmark Examples

Dynamic benchmarking system for evaluating LLM models on HuggingFace datasets using Braintrust.

## Setup

1. Install dependencies:
```bash
pip install braintrust autoevals openai
```

2. Get your Braintrust API key:
   - Sign up at [braintrust.dev](https://braintrust.dev)
   - Go to Settings → API Keys
   - Create a new API key

3. Set environment variables:
```bash
export BRAINTRUST_API_KEY=your_key_here
```

4. Configure LLM provider keys in Braintrust:
   - This project uses the Braintrust AI proxy to route requests to different LLM providers
   - Add your OpenAI, Anthropic, and other provider API keys to your Braintrust account settings
   - The proxy will automatically route requests based on the model name

## Usage

1. Load datasets:
```bash
python data/load_data.py
python data/push_data_braintrust.py
```

2. Run evaluations:
```bash
python evals/benchmark_eval.py
```

3. Control evaluation scope:
```bash
export EVAL_MODELS="gpt-4o-mini,claude-3-haiku-20240307"
export EVAL_DATASETS="HuggingFaceH4/aime/2024"
python evals/benchmark_eval.py
```

## Features

- Dynamic dataset discovery
- Multiple LLM model support via Braintrust AI proxy
- Factuality scoring with autoevals
- Automated result tracking in Braintrust
