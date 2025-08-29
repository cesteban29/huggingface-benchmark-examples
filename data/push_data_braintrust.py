import braintrust
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

def discover_datasets(data_dir: str = "data") -> Dict[str, List[str]]:
    """
    Discover all downloaded HuggingFace datasets in the data directory.
    Returns a dictionary mapping dataset names to their JSON file paths.
    """
    datasets = {}
    data_path = Path(data_dir)
    
    # Skip the script files
    ignore_files = {"load_data.py", "push_data_braintrust.py"}
    
    for item in data_path.iterdir():
        if item.is_dir():
            # Look for JSON files in each dataset directory
            json_files = list(item.glob("*.json"))
            if json_files:
                dataset_name = item.name
                datasets[dataset_name] = [str(f) for f in json_files]
                print(f"Found dataset: {dataset_name} with {len(json_files)} split(s)")
    
    return datasets

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def push_dataset_to_braintrust(
    dataset_name: str, 
    json_files: List[str],
    project_name: str = "HuggingFace Benchmarks",
    batch_size: int = 100
) -> None:
    """
    Push a dataset to Braintrust.
    
    Args:
        dataset_name: Name of the dataset (e.g., "evalplus_humanevalplus")
        json_files: List of JSON file paths for this dataset
        project_name: Braintrust project name
        batch_size: Number of records to insert in a single batch
    """
    # Initialize Braintrust dataset
    bt_dataset = braintrust.init_dataset(
        project=project_name, 
        name=dataset_name.replace('_', '/')  # Convert back to original format
    )
    print(f"\nPushing dataset: {dataset_name} to Braintrust project: {project_name}")
    
    total_records = 0
    
    for json_file in json_files:
        split_name = Path(json_file).stem  # e.g., "train", "test", "validation"
        print(f"  Processing split: {split_name}")
        
        # Load the data
        data = load_json_data(json_file)
        print(f"    Loaded {len(data)} records from {json_file}")
        
        # Insert data in batches for better performance
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            for record in batch:
                # Prepare the record for Braintrust
                # The structure depends on the dataset format
                
                # For HumanEvalPlus, typical fields might be: task_id, prompt, canonical_solution, test, etc.
                # For AIME 2024, fields might be: problem, solution, answer, etc.
                
                # Generic approach: use the entire record as input
                # You can customize this based on specific dataset structures
                
                input_data = {}
                expected_data = {}
                metadata = {"split": split_name}
                
                # Try to intelligently split fields between input and expected
                for key, value in record.items():
                    if key in ["prompt", "problem", "question", "input", "instruction"]:
                        input_data[key] = value
                    elif key in ["solution", "canonical_solution", "answer", "output", "response", "test"]:
                        expected_data[key] = value
                    else:
                        # Add other fields to metadata
                        metadata[key] = value
                
                # If no clear input/expected split, use the whole record as input
                if not input_data:
                    input_data = record
                
                # Insert the record
                bt_dataset.insert(
                    input=input_data,
                    expected=expected_data if expected_data else None,
                    metadata=metadata
                )
            
            total_records += len(batch)
            print(f"    Inserted batch: {len(batch)} records (total: {total_records})")
    
    print(f"  ✓ Successfully pushed {total_records} records to Braintrust")
    
    # Finalize the dataset
    bt_dataset.summarize()

def main():
    """Main function to discover and push all datasets to Braintrust."""
    print("="*60)
    print("BRAINTRUST DATASET UPLOADER")
    print("="*60)
    
    # Discover all datasets
    datasets = discover_datasets()
    
    if not datasets:
        print("\nNo datasets found in the data directory.")
        print("Please run load_data.py first to download datasets from HuggingFace.")
        return
    
    print(f"\nFound {len(datasets)} dataset(s) to push to Braintrust")
    
    # Configuration
    project_name = os.getenv("BRAINTRUST_PROJECT", "HuggingFace Benchmarks")
    
    # Push each dataset
    for dataset_name, json_files in datasets.items():
        try:
            push_dataset_to_braintrust(
                dataset_name=dataset_name,
                json_files=json_files,
                project_name=project_name
            )
        except Exception as e:
            print(f"\n✗ Error pushing {dataset_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("UPLOAD COMPLETE")
    print("="*60)
    print(f"All datasets have been processed. Check your Braintrust project: {project_name}")

if __name__ == "__main__":
    main()

