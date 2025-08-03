
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskPrototype:

    task_id: str
    language: str
    sample_size: int
    
    input_embedding: np.ndarray
    output_embedding: np.ndarray
    
    prototype_embedding: np.ndarray
    
    # Metadata
    creation_timestamp: str
    sample_examples: List[Dict]


class TaskPrototypeExtractor:

    def __init__(self, 
                 embedding_model: str = "microsoft/codebert-base",
                 max_sample_size: int = 500,
                 min_sample_size: int = 200):

        self.embedding_model = embedding_model
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model)
            self.model.eval()
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            raise RuntimeError(f"Cannot initialize without embedding model: {e}")
    
    def _compute_semantic_embedding(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)
        
        return np.mean(embeddings, axis=0)
    
    def extract_prototype(self, 
                         task_id: str,
                         sample_data: List[Dict],
                         language: str) -> TaskPrototype:
   
        logger.info(f"Extracting prototype for task: {task_id}")
        
        # Validate input
        if len(sample_data) < self.min_sample_size:
            raise ValueError(f"Insufficient samples: {len(sample_data)} < {self.min_sample_size}")
        
        if language is None:
            raise ValueError("Language must be provided")
        
        if len(sample_data) > self.max_sample_size:
            sample_data = sample_data[:self.max_sample_size]
        
        inputs = [item['input'] for item in sample_data]
        outputs = [item['output'] for item in sample_data]
        
        logger.info(f"Processing {len(sample_data)} samples for language: {language}")
        
        input_embedding = self._compute_semantic_embedding(inputs)
        output_embedding = self._compute_semantic_embedding(outputs)
        
        # Combine input and output embeddings for final prototype
        prototype_embedding = np.concatenate([input_embedding, output_embedding])
        
        # Create prototype
        from datetime import datetime
        prototype = TaskPrototype(
            task_id=task_id,
            language=language,
            sample_size=len(sample_data),
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            prototype_embedding=prototype_embedding,
            creation_timestamp=datetime.now().isoformat(),
            sample_examples=sample_data[:5]  # Store first 5 examples
        )
        
        logger.info(f"Prototype extracted: {language} task with embedding shape {prototype_embedding.shape}")
        return prototype
    
    def compute_prototype_similarity(self, 
                                   prototype1: TaskPrototype, 
                                   prototype2: TaskPrototype) -> float:
        """
        Compute similarity between two task prototypes using cosine similarity
        
        Args:
            prototype1: First task prototype
            prototype2: Second task prototype
            
        Returns:
            Similarity score between 0 and 1
        """
        # Language must match for meaningful similarity
        if prototype1.language != prototype2.language:
            return 0.0
        
        # Compute cosine similarity between prototype embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            prototype1.prototype_embedding.reshape(1, -1),
            prototype2.prototype_embedding.reshape(1, -1)
        )[0][0]
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    extractor = TaskPrototypeExtractor()
    
    # Sample data for testing
    python_samples = [
        {"input": "[LANGUAGE] python [COMMAND] Write a function to add two numbers", 
         "output": "def add(a, b):\n    return a + b"},
        {"input": "[LANGUAGE] python [COMMAND] Create a function to multiply two values", 
         "output": "def multiply(x, y):\n    return x * y"},
        {"input": "[LANGUAGE] python [COMMAND] Write a function to calculate factorial", 
         "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
    ]
    
    try:
        prototype = extractor.extract_prototype("python_math_functions", python_samples, "python")
        print(f"Extracted prototype for {prototype.task_id}")
        print(f"Language: {prototype.language}")
        print(f"Sample size: {prototype.sample_size}")
        print(f"Input embedding shape: {prototype.input_embedding.shape}")
        print(f"Output embedding shape: {prototype.output_embedding.shape}")
        print(f"Prototype embedding shape: {prototype.prototype_embedding.shape}")
    except Exception as e:
        print(f"Error: {e}")
