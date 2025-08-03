
import re
import logging
from typing import List, Dict, Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def contains_url(text: str) -> bool:
    url_pattern = re.compile(r'(https?://|www\.)[^\s/$.?#].[^\s]*', re.IGNORECASE)
    return bool(url_pattern.search(text))


def filtering_rules(example: Dict) -> bool:
    doc = example["docstring"].strip()
    code = example["code"].strip()

    if not (30 <= len(doc) <= 300 and 30 <= len(code) <= 600):
        return False
    
    if contains_url(doc):
        return False
    
    if "self" in code:
        return False
    
    return True


def normalize_example(example: Dict) -> Dict:
    lang = example.get("language", "unknown")
    input_text = example["docstring"].strip()
    
    # Format input with language context
    formatted_input = f"[LANGUAGE] {lang} [COMMAND] {input_text}"
    
    return {
        "input": formatted_input,
        "output": example["code"].strip(),
        "language": lang  
    }


def load_vault_dataset(
    split_set: List[str] = ["train/small"],
    languages: List[str] = ['python', 'java', 'c', 'rust', 'ruby', 'go'],
    trust_remote_code: bool = True,
    apply_filtering: bool = False
) -> Union[Dataset, DatasetDict]:
    logger.info(f"Loading Vault dataset with splits: {split_set}, languages: {languages}")
    
    try:
        # Load dataset
        dataset = load_dataset(
            "Fsoft-AIC/the-vault-function",
            split_set=split_set,
            languages=languages,
            trust_remote_code=trust_remote_code
        )
        
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
        
        # Apply filtering if requested
        if apply_filtering:
            logger.info("Applying filtering rules...")
            original_size = len(dataset)
            dataset = dataset.filter(filtering_rules)
            filtered_size = len(dataset)
            logger.info(f"Filtered dataset: {original_size} -> {filtered_size} examples")
        
        # Normalize the dataset
        logger.info("Normalizing dataset...")
        columns_to_remove = [
            "hexsha", "repo", "path", "license", "identifier", "return_type",
            "original_string", "original_docstring", "docstring_tokens", 
            "code_tokens", "short_docstring", "short_docstring_tokens",
            "comment", "parameters", "docstring_params"
        ]
        
        # Only remove columns that actually exist
        existing_columns = dataset.column_names if hasattr(dataset, 'column_names') else []
        columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
        
        normalized_dataset = dataset.map(
            normalize_example,
            remove_columns=columns_to_remove
        )
        
        # Extract single dataset if it's a DatasetDict with one key
        if isinstance(normalized_dataset, DatasetDict):
            normalized_dataset = normalized_dataset[list(normalized_dataset.keys())[0]]
        
        logger.info(f"Dataset normalized. Available languages: {set(normalized_dataset['language'])}")
        
        return normalized_dataset
        
    except Exception as e:
        logger.error(f"Error loading Vault dataset: {e}")
        raise


def extract_dataset(dataset: Union[Dataset, DatasetDict]) -> Dataset:
    """
    Extract Dataset from DatasetDict if needed
    
    Args:
        dataset: Input dataset (Dataset or DatasetDict)
        
    Returns:
        Dataset object
    """
    if isinstance(dataset, DatasetDict):
        return dataset[list(dataset.keys())[0]]
    return dataset


def load_and_split_datasets(
    train_split_set: List[str] = ["train/small"],
    test_split_set: List[str] = ["test"],
    languages: List[str] = ['python', 'java', 'c', 'rust', 'ruby', 'go'],
    apply_filtering: bool = False
) -> Tuple[Dataset, Dataset]:
    logger.info("Loading training and test datasets...")
    
    # Load training dataset
    train_dataset = load_vault_dataset(
        split_set=train_split_set,
        languages=languages,
        apply_filtering=apply_filtering
    )
    
    # Load test dataset
    test_dataset = load_vault_dataset(
        split_set=test_split_set,
        languages=languages,
        apply_filtering=apply_filtering
    )
    
    # Extract datasets if they're DatasetDict
    train_dataset = extract_dataset(train_dataset)
    test_dataset = extract_dataset(test_dataset)
    
    logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def initialize_tokenizer(model_name: str = "t5-small") -> AutoTokenizer:
    logger.info(f"Initializing tokenizer for {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def preprocess_function(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
    padding: str = "max_length"
) -> Dict:
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=padding
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=padding
    ).input_ids
    
    # Replace pad tokens with -100 for loss calculation
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    
    # Preserve language field for continual learning
    if "language" in examples:
        model_inputs["language"] = examples["language"]
    
    return model_inputs


def tokenize_datasets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
    batch_size: int = 1000
) -> Tuple[Dataset, Dataset]:
    logger.info("Tokenizing datasets...")
    
    def preprocess_batch(examples):
        return preprocess_function(
            examples, 
            tokenizer, 
            max_input_length, 
            max_target_length
        )
    
    # Tokenize training dataset
    logger.info("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=train_dataset.column_names
    )
    
    # Tokenize test dataset
    logger.info("Tokenizing test dataset...")
    tokenized_test = test_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=test_dataset.column_names
    )
    
    logger.info(f"Tokenization complete - Train: {len(tokenized_train)}, Test: {len(tokenized_test)}")
    logger.info(f"Tokenized columns: {tokenized_train.column_names}")
    
    return tokenized_train, tokenized_test


def prepare_data_pipeline(
    train_split_set: List[str] = ["train/small"],
    test_split_set: List[str] = ["test"],
    languages: List[str] = ['python', 'java', 'c', 'rust', 'ruby', 'go'],
    model_name: str = "t5-small",
    max_input_length: int = 256,
    max_target_length: int = 256,
    apply_filtering: bool = False
) -> Tuple[Dataset, Dataset, AutoTokenizer]:

    logger.info("Starting complete data preparation pipeline...")
    
    # Load datasets
    train_dataset, test_dataset = load_and_split_datasets(
        train_split_set=train_split_set,
        test_split_set=test_split_set,
        languages=languages,
        apply_filtering=apply_filtering
    )
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_name)
    
    # Tokenize datasets
    tokenized_train, tokenized_test = tokenize_datasets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    logger.info("Data preparation pipeline completed successfully!")
    
    # Print summary
    print(f"\n{'='*50}")
    print("DATA PREPARATION SUMMARY")
    print(f"{'='*50}")
    print(f"Training samples: {len(tokenized_train)}")
    print(f"Test samples: {len(tokenized_test)}")
    print(f"Languages: {languages}")
    print(f"Max input length: {max_input_length}")
    print(f"Max target length: {max_target_length}")
    print(f"Tokenizer: {model_name}")
    print(f"Columns: {tokenized_train.column_names}")
    print(f"{'='*50}\n")
    
    return tokenized_train, tokenized_test, tokenizer


# Example usage
if __name__ == "__main__":
    # Example: Load and preprocess data
    try:
        tokenized_train, tokenized_test, tokenizer = prepare_data_pipeline(
            languages=['python', 'java', 'c'],
            apply_filtering=False
        )
        
        # Example: Check a sample
        print("Sample from training data:")
        sample = tokenized_train[0]
        print(f"Input IDs shape: {len(sample['input_ids'])}")
        print(f"Labels shape: {len(sample['labels'])}")
        print(f"Language: {sample.get('language', 'Not preserved')}")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
