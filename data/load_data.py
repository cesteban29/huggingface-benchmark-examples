from datasets import load_dataset
import json
import os
from typing import Dict, List, Any

def download_dataset_to_json(dataset_name: str, output_dir: str) -> Dict[str, str]:
    """
    Download a HuggingFace dataset and save each split as JSON.
    Returns a dictionary mapping split names to file paths.
    """
    # Create directory for this dataset
    dataset_dir = os.path.join(output_dir, dataset_name.replace('/', '_'))
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Output directory: {dataset_dir}")
    
    output_files = {}
    
    try:
        # Load the dataset to check available splits
        dataset = load_dataset(dataset_name)
        available_splits = list(dataset.keys())
        print(f"Available splits: {available_splits}")
        
        for split in available_splits:
            # Convert to list of dictionaries
            data = dataset[split].to_list()
            
            # Save to JSON
            output_file = os.path.join(dataset_dir, f"{split}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            output_files[split] = output_file
            print(f"  ✓ Saved {split} split: {len(data)} examples → {output_file}")
            
    except Exception as e:
        print(f"  ✗ Error processing dataset: {e}")
    
    return output_files

def main():
    # Define the datasets to download
    datasets = [
        "evalplus/humanevalplus",
        "HuggingFaceH4/aime_2024"
    ]
    
    # Create main data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Track all downloaded files
    all_downloads = {}
    
    for dataset_name in datasets:
        try:
            output_files = download_dataset_to_json(dataset_name, data_dir)
            all_downloads[dataset_name] = output_files
        except Exception as e:
            print(f"\nFailed to download {dataset_name}: {e}")
            all_downloads[dataset_name] = {}
    
    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    for dataset_name, files in all_downloads.items():
        if files:
            print(f"\n{dataset_name}:")
            for split, filepath in files.items():
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"  - {split}: {filepath} ({file_size:.2f} MB)")
        else:
            print(f"\n{dataset_name}: Failed to download")
    
    print("\nAll datasets saved in JSON format, ready for Braintrust SDK integration.")

if __name__ == "__main__":
    main() 