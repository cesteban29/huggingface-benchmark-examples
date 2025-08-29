from braintrust import Eval, init_dataset, wrap_openai
from autoevals import Factuality, ClosedQA, Battle
import os
from typing import Dict, Any, List, Callable
from pathlib import Path
from openai import OpenAI

project_name = os.getenv("BRAINTRUST_PROJECT", "HuggingFace Benchmarks")

models = [
    "gpt-4o-mini",
    "gpt-5-mini"
]

client = None

def get_braintrust_client():
    """
    Get Braintrust AI proxy client that supports multiple providers.
    """
    global client
    if client is None:
        client = wrap_openai(
            OpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                api_key=os.getenv("BRAINTRUST_API_KEY", os.getenv("OPENAI_API_KEY"))
            )
        )
    return client

def discover_braintrust_datasets() -> List[str]:
    """
    Discover all available datasets in Braintrust.
    This assumes datasets have been pushed using push_data_braintrust.py
    """
    data_dir = Path("data")
    datasets = []
    
    ignore_files = {"load_data.py", "push_data_braintrust.py"}
    
    for item in data_dir.iterdir():
        if item.is_dir():
            json_files = list(item.glob("*.json"))
            if json_files:
                dataset_name = item.name.replace('_', '/')
                datasets.append(dataset_name)
                print(f"Found dataset: {dataset_name}")
    
    return datasets

def create_task_function(model_name: str) -> Callable:
    """
    Create a task function for a specific model using Braintrust AI proxy.
    """
    def task(input_data: Dict[str, Any]) -> str:
        try:
            if isinstance(input_data, dict):
                if "prompt" in input_data:
                    prompt = input_data["prompt"]
                elif "problem" in input_data:
                    prompt = input_data["problem"]
                elif "question" in input_data:
                    prompt = input_data["question"]
                elif "instruction" in input_data:
                    prompt = input_data["instruction"]
                elif "input" in input_data:
                    prompt = input_data["input"]
                else:
                    prompt = str(input_data)
            else:
                prompt = str(input_data)
            
            client = get_braintrust_client()
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error calling {model_name}: {e}")
            return f"Error: {str(e)}"
    
    return task


def run_evaluation(model_name: str, dataset_name: str):
    """
    Run evaluation for a specific model and dataset.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running evaluation: {model_name} on {dataset_name}")
        print(f"{'='*60}")
        
        dataset = init_dataset(project=project_name, name=dataset_name)
        
        task_function = create_task_function(model_name)
        
        eval_name = f"{model_name}_{dataset_name.replace('/', '_')}"
        
        result = Eval(
            project_name,
            data=dataset,
            task=task_function,
            scores=[Factuality],
            experiment_name=eval_name,
            metadata={
                "model": model_name,
                "dataset": dataset_name
            }
        )
        
        print(f"✓ Completed evaluation for {model_name} on {dataset_name}")
        if hasattr(result, 'summary') and hasattr(result.summary, 'scores'):
            factuality_score = result.summary.scores.get('Factuality')
            if factuality_score and hasattr(factuality_score, 'score'):
                print(f"  Factuality Score: {factuality_score.score}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error running evaluation for {model_name} on {dataset_name}: {e}")
        return None

def main():
    """
    Main function to run all evaluations dynamically.
    """
    print("="*60)
    print("DYNAMIC BENCHMARK EVALUATION")
    print("="*60)
    
    datasets = discover_braintrust_datasets()
    
    if not datasets:
        print("\nNo datasets found. Please run the following scripts first:")
        print("1. python data/load_data.py - to download datasets from HuggingFace")
        print("2. python data/push_data_braintrust.py - to push datasets to Braintrust")
        return
    
    print(f"\nFound {len(datasets)} dataset(s)")
    print(f"Will evaluate {len(models)} model(s)")
    
    selected_models = os.getenv("EVAL_MODELS", "").split(",") if os.getenv("EVAL_MODELS") else models
    selected_datasets = os.getenv("EVAL_DATASETS", "").split(",") if os.getenv("EVAL_DATASETS") else datasets
    
    selected_models = [m.strip() for m in selected_models if m.strip()]
    selected_datasets = [d.strip() for d in selected_datasets if d.strip()]
    
    if not selected_models:
        selected_models = models
    if not selected_datasets:
        selected_datasets = datasets
    
    print(f"\nRunning evaluations for:")
    print(f"  Models: {', '.join(selected_models)}")
    print(f"  Datasets: {', '.join(selected_datasets)}")
    
    results = []
    
    for dataset_name in selected_datasets:
        for model_name in selected_models:
            result = run_evaluation(model_name, dataset_name)
            if result:
                results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "result": result
                })
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for res in results:
        print(f"\n{res['model']} on {res['dataset']}:")
        if res['result'] and hasattr(res['result'], 'summary'):
            summary = res['result'].summary
            if hasattr(summary, 'scores'):
                factuality_score = summary.scores.get('Factuality')
                if factuality_score and hasattr(factuality_score, 'score'):
                    print(f"  Factuality Score: {factuality_score.score}")
            if hasattr(summary, 'metrics'):
                duration = summary.metrics.get('duration', 'N/A')
                print(f"  Duration: {duration}ms")
    
    print("\n" + "="*60)
    print("ALL EVALUATIONS COMPLETE")
    print("="*60)
    print(f"Check your Braintrust project: {project_name}")

if __name__ == "__main__":
    main()