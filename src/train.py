
import torch
import numpy as np
from typing import Dict, List
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType
from datasets import Dataset
import logging
from datetime import datetime
import json
import os

# Import our custom modules
from data_processing import prepare_data_pipeline, load_and_split_datasets
from architecture import TaskPrototypeExtractor
from lora_router import LoRARouter


os.environ["TRANSFORMERS_NO_TF"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainingPipeline:
    """
    Sequential LoRA Task Routing Pipeline
    
    This pipeline processes programming language tasks one by one:
    1. Each task (language) is processed sequentially
    2. Task embeddings are compared with existing LoRA blocks  
    3. Best matching block is selected or new block is created
    4. ONLY the selected block is trained on that specific task
    5. No bulk training - each task trains its assigned block individually
    """
    
    def __init__(self,
                 model_name: str = "t5-small",
                 languages: List[str] = ['python', 'java', 'c', 'rust'],
                 similarity_threshold: float = 0.6,
                 max_lora_blocks: int = 8,
                 output_dir: str = "./lora_routing_outputs"):
 
        self.model_name = model_name
        self.languages = languages
        self.similarity_threshold = similarity_threshold
        self.max_lora_blocks = max_lora_blocks
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.base_model = None
        self.tokenizer = None
        self.prototype_extractor = None
        self.lora_router = None
        
        # Training data
        self.train_dataset = None
        self.test_dataset = None
        self.language_datasets = None
        
        logger.info(f"Initialized LoRA Training Pipeline for languages: {languages}")
        
        # Print GPU info if available
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("No GPU available, using CPU")
    
    def check_gpu_memory(self):
        """Check GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def setup_base_model(self):
        """Load and setup the base model and tokenizer"""
        logger.info(f"Loading base model: {self.model_name}")
        
        # Load model and tokenizer
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Base model and tokenizer loaded successfully on {self.device}")
    
    def load_and_prepare_data(self,
                            train_split_set: List[str] = ["train/small"],
                            test_split_set: List[str] = ["test"],
                            max_input_length: int = 256,
                            max_target_length: int = 256):
        """Load and prepare training data"""
        logger.info("Loading and preparing datasets...")
        
        # Load tokenized datasets
        self.train_dataset, self.test_dataset, _ = prepare_data_pipeline(
            train_split_set=train_split_set,
            test_split_set=test_split_set,
            languages=self.languages,
            model_name=self.model_name,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            # include_language_in_input=True,
            apply_filtering=False
        )
        
        # Load raw datasets for prototype extraction (already language-specific)
        train_raw, test_raw = load_and_split_datasets(
            train_split_set=train_split_set,
            test_split_set=test_split_set,
            languages=self.languages,
            # include_language_in_input=True
        )
        
        # Initialize language_datasets as empty dictionary
        self.language_datasets = {}
        
        for lang in self.languages:
            # Filter for this specific language
            lang_data = train_raw.filter(lambda x: x.get('language').lower() == lang.lower())
            if len(lang_data) > 0:
                self.language_datasets[lang] = lang_data
                logger.info(f"Language dataset {lang}: {len(lang_data)} examples")
            else:
                logger.warning(f"No examples found for language: {lang}")
        
        logger.info(f"Data prepared - Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
        logger.info(f"Total languages with data: {len(self.language_datasets)}")
    
    def setup_prototype_extractor_and_router(self):
        """Initialize task prototype extractor and LoRA router"""
        logger.info("Setting up prototype extractor and LoRA router...")
        
        # Initialize prototype extractor
        self.prototype_extractor = TaskPrototypeExtractor(
            embedding_model="microsoft/codebert-base",
            max_sample_size=50,
            min_sample_size=10
        )
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # T5 specific
        )
        
        # Initialize LoRA router
        self.lora_router = LoRARouter(
            base_model=self.base_model,
            prototype_extractor=self.prototype_extractor,
            similarity_threshold=self.similarity_threshold,
            max_blocks=self.max_lora_blocks,
            lora_config=lora_config
        )
        
        logger.info(f"Prototype extractor and LoRA router initialized on {self.device}")
    
    def process_and_train_task(self, language: str, sample_size: int = 300):
        """Process a single task (language) and train the assigned block"""
        logger.info(f"Processing task: {language}")
        
        # Get dataset for this language
        if language not in self.language_datasets:
            logger.error(f"No dataset available for {language}")
            return None
            
        dataset = self.language_datasets[language]
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for {language}")
            return None
            
        logger.info(f"Dataset size for {language}: {len(dataset)} examples")
        
        # Sample data for prototype extraction
        actual_sample_size = min(sample_size, len(dataset))
        indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
        # Convert numpy indices to Python integers for HuggingFace Dataset compatibility
        sample_data = [dataset[int(i)] for i in indices]
        
        # Create task ID
        task_id = f"{language}_code_generation_task"
        
        try:
            # Route the task - this will either reuse existing block or create new one
            block_id, decision, similarity, prototype = self.lora_router.route_task(
                task_id=task_id,
                sample_data=sample_data,
                language=language
            )
            
            logger.info(f"Task {task_id} routed to block {block_id} (decision: {decision}, similarity: {similarity:.3f})")
            
            # Train ONLY this block with data for this task
            training_result = self.train_single_task_on_block(
                block_id=block_id,
                language=language,
                task_id=task_id
            )
            
            return {
                'task_id': task_id,
                'language': language,
                'block_id': block_id,
                'routing_decision': decision,
                'similarity': similarity,
                'training_result': training_result,
                'sample_size': actual_sample_size
            }
            
        except Exception as e:
            logger.error(f"Error processing task {language}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'task_id': task_id,
                'language': language,
                'error': str(e)
            }
    
    def train_single_task_on_block(self,
                                   block_id: str,
                                   language: str,
                                   task_id: str,
                                   num_epochs: int = 2,
                                   batch_size: int = 4,
                                   learning_rate: float = 5e-4):
        """Train a specific LoRA block on a single task"""
        logger.info(f"Training block {block_id} on task {task_id} ({language})")
        
        # Get the model for this block
        model = self.lora_router.get_block_model(block_id)
        
        # Prepare training data for this specific language only
        def is_target_language(example):
            return example.get('language').lower() == language.lower()
        
        task_train_data = self.train_dataset.filter(is_target_language)
        
        if len(task_train_data) == 0:
            logger.warning(f"No training data for {language}, skipping training")
            return None
            
        logger.info(f"Training on {len(task_train_data)} examples for {language}")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{block_id}_{language}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            remove_unused_columns=True,  # This will remove non-model columns like 'language'
            dataloader_pin_memory=False,
            report_to=None,
            # Performance optimizations
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=2 if torch.cuda.is_available() else 0,
            group_by_length=True,  # Group sequences by length for efficiency
        )
        
        # Data collator with truncation enabled
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=task_train_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info(f"Starting training for {block_id} on {language}...")
        
        # Check GPU memory before training
        if torch.cuda.is_available():
            self.check_gpu_memory()
        
        train_result = trainer.train()
        
        # Save the trained model
        trainer.save_model(f"{self.output_dir}/{block_id}_{language}_final")
        
        logger.info(f"Training completed for {block_id} on {language}. Loss: {train_result.training_loss:.4f}")
        
        return {
            'success': True,
            'final_loss': train_result.training_loss,
            'num_examples': len(task_train_data)
        }
    
    def run_sequential_task_training(self, sample_size_per_task: int = 300):
        """Process tasks sequentially - each task gets routed and trained individually"""
        logger.info("🔄 Starting sequential task-based training")
        
        task_results = []
        
        for language in self.languages:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing Task: {language}")
            logger.info(f"{'='*50}")
            
            # Process this single task
            result = self.process_and_train_task(language, sample_size_per_task)
            
            if result:
                task_results.append(result)
                
                # Print immediate results
                if 'error' not in result:
                    logger.info(f"   Task {language} completed:")
                    logger.info(f"   Block: {result['block_id']}")
                    logger.info(f"   Routing: {result['routing_decision']}")
                    logger.info(f"   Similarity: {result['similarity']:.3f}")
                    if result['training_result']:
                        logger.info(f"   Training Loss: {result['training_result']['final_loss']:.4f}")
                else:
                    logger.error(f"Task {language} failed: {result['error']}")
            else:
                logger.error(f"Task {language} returned no result")
        
        return task_results
    
    def run_complete_pipeline(self,
                            sample_size_per_task: int = 300,
                            num_epochs: int = 2,
                            batch_size: int = 4,
                            learning_rate: float = 5e-4):
        """Run the complete sequential task training pipeline"""
        logger.info("🚀 Starting Sequential LoRA Task Routing Pipeline")
        
        try:
            # Step 1: Setup base model
            self.setup_base_model()
            
            # Step 2: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 3: Setup prototype extractor and router
            self.setup_prototype_extractor_and_router()
            
            # Step 4: Process tasks sequentially
            task_results = self.run_sequential_task_training(sample_size_per_task)
            
            # Step 5: Save results
            self.save_experiment_results(task_results)
            
            # Step 6: Print summary
            self.print_final_summary(task_results)
            
            logger.info("Sequential pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_experiment_results(self, task_results: List[Dict]):
        """Save experiment results and statistics"""
        results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'languages': self.languages,
                'similarity_threshold': self.similarity_threshold,
                'max_lora_blocks': self.max_lora_blocks,
                'pipeline_type': 'sequential_task_routing'
            },
            'task_results': task_results,
            'router_statistics': self.lora_router.get_routing_statistics() if self.lora_router else {}
        }
        
        # Save to JSON
        results_file = f"{self.output_dir}/experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save routing history
        if self.lora_router:
            self.lora_router.save_routing_history(f"{self.output_dir}/routing_history.json")
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def print_final_summary(self, task_results: List[Dict]):
        """Print final summary of the sequential task training pipeline"""
        print(f"\n{'='*80}")
        print("SEQUENTIAL LORA TASK ROUTING SUMMARY")
        print(f"{'='*80}")
        
        successful_tasks = 0
        total_tasks = len(task_results)
        
        print(f"\n📋 Task Processing Results:")
        for result in task_results:
            if 'error' not in result:
                successful_tasks += 1
                training_success = result.get('training_result', {}).get('success', False)
                loss = result.get('training_result', {}).get('final_loss', 'N/A')
                print(f" {result['language']}: Block {result['block_id']}, "
                      f"Routing: {result['routing_decision']}, "
                      f"Similarity: {result['similarity']:.3f}, "
                      f"Training: {'✓' if training_success else '✗'}, "
                      f"Loss: {loss:.4f}" if isinstance(loss, float) else f"Loss: {loss}")
            else:
                print(f"  ❌ {result['language']}: Error - {result['error']}")
        
        print(f"\n� Overall Statistics:")
        print(f"  Total tasks processed: {total_tasks}")
        print(f"  Successful tasks: {successful_tasks}/{total_tasks}")
        if total_tasks > 0:
            print(f"  Success rate: {successful_tasks/total_tasks*100:.1f}%")
        
        # Router statistics
        if hasattr(self, 'lora_router') and self.lora_router and len(self.lora_router.lora_blocks) > 0:
            print(f"  LoRA blocks created: {len(self.lora_router.lora_blocks)}")
            self.lora_router.print_routing_summary()
        else:
            print(f"  LoRA blocks created: 0")
        
        print(f"\n💾 Results saved to: {self.output_dir}")
        print(f"{'='*80}")


def main():
    """Main function to run the training pipeline"""
    # Configuration
    config = {
        'model_name': 'Salesforce/codet5-small',
        'languages': ['rust'],  # Multiple tasks to show routing
        'similarity_threshold': 0.6,  # Lower threshold for cross-language similarity
        'max_lora_blocks': 3,
        'output_dir': './lora_routing_results',
        'sample_size_per_language': 200,  # Reasonable sample size
        'num_epochs': 1,  # Quick training for testing
        'batch_size': 4,
        'learning_rate': 5e-4
    }
    
    print("🔧 Sequential LoRA Task Routing Configuration:")
    print("   Note: Tasks are processed sequentially, one at a time.")
    print("   Each task is routed to best matching block based on similarity.")
    print("   Only the selected block is trained for each task.")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize and run pipeline
    pipeline = LoRATrainingPipeline(
        model_name=config['model_name'],
        languages=config['languages'],
        similarity_threshold=config['similarity_threshold'],
        max_lora_blocks=config['max_lora_blocks'],
        output_dir=config['output_dir']
    )
    
    # Run the complete pipeline
    pipeline.run_complete_pipeline(
        sample_size_per_task=config['sample_size_per_language'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )


if __name__ == "__main__":
    main()
