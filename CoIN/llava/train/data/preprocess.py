import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Optional

class DatasetPreprocessor:
    """
    A class to load datasets from Hugging Face and save them in organized JSONL format.
    
    This class handles downloading, preprocessing, and saving datasets for various 
    code-related tasks in a structured directory format.
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize the dataset preprocessor.
        
        Args:
            data_root (str): Root directory where datasets will be saved
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Task configurations based on cl_dataset.py
        # self.task_configs = {
        #     'CONCODE_java': {
        #         'dataset_name': 'AhmedSSoliman/CodeXGLUE-CONCODE',
        #         'text_key': 'nl',
        #         'label_key': 'code',
        #         'instruction': 'Generate Java code from the following English description: ',
        #         'language': 'java'            
        #         },
        #     'CodeTrans_java_to_csharp': {
        #         'dataset_name': 'CM/codexglue_codetrans',
        #         'text_key': 'java',
        #         'label_key': 'cs',
        #         'instruction': 'Translate the following Java code into C#: ',
        #         'language': 'java_to_csharp'
        #     },
        #     'CodeSearchNet_ruby': {
        #         'dataset_name': 'semeru/code-text-ruby',
        #         'text_key': 'code',
        #         'label_key': 'docstring',
        #         'instruction': 'Summarize the following Ruby code into English: ',
        #         'language': 'ruby'
        #     },
        #     'BFP_java': {
        #         'dataset_name': 'ayeshgk/code_x_glue_cc_code_refinement_annotated',
        #         'text_key': 'buggy',
        #         'label_key': 'fixed',
        #         'instruction': 'Refactor or improve the following Java code: ',
        #         'language': 'java'
        #     }
        # }

        self.task_configs = {
            'CodeSearchNet_go': {
                'dataset_name': 'google/code_x_glue_ct_code_to_text',
                'text_key': 'code',
                'label_key': 'docstring',
                'instruction': 'Summarize the following Ruby code into English: ',
                'language': 'go'            
                },
            'CodeSearchNet_java': {
                'dataset_name': 'google/code_x_glue_ct_code_to_text',
                'text_key': 'code',
                'label_key': 'docstring',
                'instruction': 'Summarize the following Ruby code into English: ',
                'language': 'java'            
                },
            'CodeSearchNet_javascript': {
                'dataset_name': 'google/code_x_glue_ct_code_to_text',
                'text_key': 'code',
                'label_key': 'docstring',
                'instruction': 'Summarize the following Ruby code into English: ',
                'language': 'javascript'            
                
            }
        }
        
    def load_and_preprocess_task(self, 
                                task: str, 
                                include_instruction: bool = True,
                                max_samples: Optional[int] = None,
                                seed: int = 42) -> Dict[str, List[Dict]]:

        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_configs.keys())}")
        
        config = self.task_configs[task]
        print(f"Loading {task} dataset from {config['dataset_name']}...")
        
        # Load dataset splits
        splits_data = {}
        available_splits = ['train', 'validation', 'test']
        
        for split in available_splits:
            try:
                dataset = load_dataset(config['dataset_name'], config['language'], split=split)
                print(f"  Loaded {split} split: {len(dataset)} samples")
                
                # Sample if requested
                if max_samples and len(dataset) > max_samples:
                    np.random.seed(seed)
                    indices = np.random.choice(len(dataset), max_samples, replace=False)
                    dataset = dataset.select(indices)
                    print(f"  Sampled down to {len(dataset)} samples")
                
                # Preprocess the data
                processed_data = []
                for example in dataset:
                    processed_example = self._preprocess_example(
                        example, config, include_instruction
                    )
                    processed_data.append(processed_example)
                
                splits_data[split] = processed_data
                
            except Exception as e:
                print(f"  Warning: Could not load {split} split - {e}")
                continue
        
        # If no validation split, create one from train
        if 'validation' not in splits_data and 'train' in splits_data:
            print("  No validation split found, creating from train split...")
            train_data = splits_data['train']
            val_size = min(1000, len(train_data) // 10)  # 10% or 1000 samples max
            
            np.random.seed(seed)
            indices = np.random.permutation(len(train_data))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            splits_data['validation'] = [train_data[i] for i in val_indices]
            splits_data['train'] = [train_data[i] for i in train_indices]
            
            print(f"  Created validation split: {len(splits_data['validation'])} samples")
            print(f"  Updated train split: {len(splits_data['train'])} samples")
        
        return splits_data
    
    def _preprocess_example(self, 
                           example: Dict, 
                           config: Dict, 
                           include_instruction: bool) -> Dict:

        text = example[config['text_key']]
        label = example[config['label_key']]
        
        # Add instruction if requested
        if include_instruction:
            input_text = config['instruction'] + text
        else:
            input_text = text
        
        processed_example = {
            'input': input_text,
            'output': label,
            'raw_input': text,
            'instruction': config['instruction'],
            'language': config['language'],
        }
        
        
        
        return processed_example
    
    def save_task_data(self, 
                      task: str, 
                      splits_data: Dict[str, List[Dict]], 
                      format: str = 'jsonl') -> None:
        
        task_dir = self.data_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {task} data to {task_dir}...")
        
        for split_name, data in splits_data.items():
            if format == 'jsonl':
                file_path = task_dir / f"{split_name}.jsonl"
                with open(file_path, 'w', encoding='utf-8') as f:
                    for example in data:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
            elif format == 'json':
                file_path = task_dir / f"{split_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  Saved {split_name}: {len(data)} samples -> {file_path}")
        
        # Save task metadata
        metadata = {
            'task': task,
            'config': self.task_configs[task],
            'splits': {split: len(data) for split, data in splits_data.items()},
            'total_samples': sum(len(data) for data in splits_data.values()),
            'format': format
        }
        
        metadata_path = task_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved metadata -> {metadata_path}")
    
    def process_all_tasks(self, 
                         include_instruction: bool = True,
                         max_samples: Optional[int] = None,
                         format: str = 'jsonl',
                         seed: int = 42) -> None:
 
        print("Processing all tasks...")
        
        for task in self.task_configs.keys():
            try:
                print(f"\n{'='*50}")
                print(f"Processing task: {task}")
                print(f"{'='*50}")
                
                splits_data = self.load_and_preprocess_task(
                    task=task,
                    include_instruction=include_instruction,
                    max_samples=max_samples,
                    seed=seed
                )
                
                self.save_task_data(task, splits_data, format)
                
            except Exception as e:
                print(f"Error processing {task}: {e}")
                continue
        
        print(f"\n{'='*50}")
        print("Processing complete!")
        print(f"All data saved to: {self.data_root}")
    


def main():
    preprocessor = DatasetPreprocessor(data_root="data")
    
    
    preprocessor.process_all_tasks(
        include_instruction=True,
        format='jsonl'
    )
    
    # Or process individual tasks
    # for task in ['CONCODE', 'CodeTrans', 'CodeSearchNet', 'BFP']:
    #     try:
    #         splits_data = preprocessor.load_and_preprocess_task(task, max_samples=1000)
    #         preprocessor.save_task_data(task, splits_data)
    #     except Exception as e:
    #         print(f"Failed to process {task}: {e}")


if __name__ == "__main__":
    main()
