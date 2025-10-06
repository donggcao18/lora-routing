import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

import json
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(self, 
                 tokenizer,
                 task: str,
                 split: str = "train",
                 data_root: str = "data",
                 max_length: int = 512,
                 use_instruction: bool = True,
                 max_train_size: int = 15000,
                 max_val_size: int = 150,
                 max_test_size: int = 150):
        
        self.tokenizer = tokenizer
        self.task = task
        self.split = split
        self.data_root = data_root
        self.max_length = max_length
        self.use_instruction = use_instruction
        self.max_train_size = max_train_size
        self.max_val_size = max_val_size
        self.max_test_size = max_test_size

        # load examples immediately
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict]:
        if self.split == "train":
            max_size = self.max_train_size
        elif self.split == "validation":
            max_size = self.max_val_size
        elif self.split == "test":
            max_size = self.max_test_size
        else:
            max_size = -1

        data_path = Path(self.data_root) / self.task / f"{self.split}.jsonl"
        if not data_path.exists():
            print(f"Warning: JSONL not found: {data_path}")
            return []

        examples, count = [], 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_size > 0 and count >= max_size:
                    break
                try:
                    examples.append(json.loads(line.strip()))
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping bad JSON at line {line_num}: {e}")
        print(f"Loaded {len(examples)} examples from {data_path} (limit {max_size if max_size > 0 else 'unlimited'})")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._tokenize_example(self.examples[idx])

    def _tokenize_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        input_text = example["input"] if self.use_instruction else example["raw_input"]
        output_text = example["output"]

        # EOS can be added depending on tokenizer config; optional here
        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,            # let DataCollator handle padding
        )
        target_enc = self.tokenizer(
            output_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return {
            "input_ids": input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels": target_enc["input_ids"]
        }

def get_dataloader(tokenizer,
                   task: str,
                #    split: str = "train",
                   data_root: str = "data",
                   batch_size: int = 8,
                   max_length: int = 512,
                   use_instruction: bool = True,
                   max_train_size: int = 15000,
                   max_val_size: int = 150,
                   max_test_size: int = 150,
                   shuffle: bool = True) -> DataLoader:
    """
    Factory function to create a DataLoader for a given task + split.
    """
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    array = []
    for split in ["train", "validation", "test"]:
        dataset = T5Dataset(
            tokenizer=tokenizer,
            task=task,
            split=split,
            data_root=data_root,
            max_length=max_length,
            use_instruction=use_instruction,
            max_train_size=max_train_size,
            max_val_size=max_val_size,
            max_test_size=max_test_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
            num_workers=0,   # safe default for Kaggle
            pin_memory=True
        )
        array.append(dataloader)
    
    return array

def get_Dataset(task, 
                 split,
                 data_root,
                 max_length,
                 use_instruction,
                 max_test_size,
                 max_val_size,
                 max_train_size):
    #Fill out the model's name inn this field

    tok = AutoTokenizer("")
    tempp = T5Dataset(tok,
                 task,
                 split,
                 data_root,
                 max_length,
                 use_instruction,
                 max_train_size,
                 max_val_size,
                 max_test_size)
    