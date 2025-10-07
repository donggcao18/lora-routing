

import os
import json
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys

# from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from llava.train.transformers.trainer import Trainer

from ..model import *

from llava.train.CoIN.peft import  TaskType, get_peft_model, LoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
import logging
from datetime import datetime

from llava.train.eval import evaluate_model_on_task
from llava.train.data.cl_dataset import T5Dataset

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb completely

#IO control
input_dir = "/kaggle/input/codetask-extend/data"
output_dir = "/kaggle/working/"

log_filename = f"{output_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")


class LoRATrainingPipeline:

    
    def __init__(self,
                 model_name: str = "t5-small",
                 output_dir: str = "data",
                 input_dir: str = "lora_routing_results",
                 num_epochs: int = 1,
                 batch_size: int = 4,
                 gradient_accumulation_steps: int = 2,
                 learning_rate: float = 3e-4,
                 repetition_penalty: float = 1.2,
                 task_sequence: List[str] = None,
                 max_length: int = 512,
                 max_train_size: int = 10000,
                 max_val_size: int = 150,
                 max_test_size: int = 150):

        self.model_name = model_name
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.max_train_size = max_train_size
        self.max_val_size = max_val_size
        self.max_test_size = max_test_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize components
        self.base_model = None
        self.tokenizer = None
        self.lora_model = None  # Single LoRA model for all tasks
        self.trainer = None     # Single trainer for all tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Data
        self.task_sequence = task_sequence if task_sequence else []
        self.task_list = {}

        
        # Print GPU info if available
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("No GPU available, using CPU")
    
    
    def setup_base_model(self):
        """Load and setup the base model and tokenizer"""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = self.base_model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Base model and tokenizer loaded successfully on {self.device}")
    
    def get_tasks_data_dict(self, memory_perc=0):
        tasks_data_dict = {}

        for task in self.task_list:
            print(f"\nLoading data for task: {task}")
            tasks_data_dict[task] = {}
            for split in ["train", "validation", "test"]:
                tasks_data_dict[task][split] = T5Dataset(self.tokenizer,
                                                                task=task,
                                                                split=split,
                                                                data_root=self.input_dir,
                                                                max_length=self.max_length,
                                                                max_train_size=self.max_train_size,
                                                                max_val_size=self.max_val_size,
                                                                max_test_size=self.max_test_size
                                                            )
            
            if memory_perc:
                logger.info(f"Reducing memory usage for {task} dataloaders by {memory_perc*100:.1f}%")
                #may process this part later

        
        return tasks_data_dict
    
    def create_lora_config(self):
        """Create LoRA configuration"""
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q", "v"],  # T5 attention modules
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            dema = "True",
        )
        logger.info("LoRA configuration created")
        return lora_config

    def setup_lora_model(self):
        
        if self.base_model is None:
            self.setup_base_model()
        
        lora_config = self.create_lora_config()
        self.lora_model = get_peft_model(self.base_model, lora_config)
        self.lora_model.print_trainable_parameters()
        
        logger.info("LoRA model setup completed")
        return self.lora_model

    def setup_trainer(self):
        """Create a single trainer for sequential training"""
        if self.lora_model is None:
            self.setup_lora_model()
        
        # Training arguments for sequential learning
        # training_args = TrainingArguments(
        #     output_dir=f"{self.output_dir}/lora_checkpoints/sequential_training",
        #     num_train_epochs=self.num_epochs,
        #     per_device_train_batch_size=self.batch_size,
        #     per_device_eval_batch_size=self.batch_size,
        #     warmup_steps=100,
        #     weight_decay=0.01,
        #     logging_dir=f"{self.output_dir}/logs/sequential_training",
        #     logging_steps=50,
        #     save_strategy="epoch",
        #     evaluation_strategy="epoch",
        #     load_best_model_at_end=False,
        #     metric_for_best_model="eval_loss",
        #     greater_is_better=False,
        #     report_to=[],  # Disable wandb
        #     gradient_accumulation_steps=self.gradient_accumulation_steps,
        #     dataloader_pin_memory=False,
        #     learning_rate=self.learning_rate,
        # )

        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/lora_checkpoints/sequential_training",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs/sequential_training",
            logging_steps=50,
            save_strategy="no",
            evaluation_strategy="epoch",
            load_best_model_at_end=False,
            greater_is_better=False,
            report_to=[],  # Disable wandb
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_pin_memory=False,
            learning_rate=self.learning_rate,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.lora_model,
            padding="max_length",
            max_length=512, 
            return_tensors="pt"
        )
        
        # Create the trainer
        self.trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info("Single trainer created for sequential training")

    
        
    def train_on_task(self, task_name: str, train_dataset, val_dataset):
        """Train LoRA on a specific task using the single persistent trainer"""
        logger.info(f"Training LoRA on task: {task_name}")
        
        # Ensure trainer is set up
        if self.trainer is None:
            self.setup_trainer()
        
        # Update the trainer's datasets for this task
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = val_dataset
        
        # Train - this continues training the same LoRA parameters
        logger.info(f"Starting continual training for {task_name}...")
        logger.info(f"LoRA model will accumulate knowledge from this task")
        
        # Reset the trainer's state for new task (but keep the model)
        self.trainer.state.epoch = 0
        self.trainer.state.global_step = 0
        
        self.trainer.train()
        
        logger.info(f"Training completed for {task_name}. LoRA parameters updated.")
        
        


    def evaluate_on_all_tasks(self, tasks_data_dict: Dict) -> Dict:
        """Evaluate current LoRA model on validation sets of all tasks using BLEU scores"""
        logger.info("Evaluating LoRA model on all task validation sets...")
        
        results = {}
        self.lora_model.eval()
        
        for task_name, task_data in tasks_data_dict.items():
            logger.info(f"Evaluating on {task_name} validation set...")
            
            # Get validation dataset
            val_dataset = task_data['validation']
            
            # Use evaluate_model_on_task function to get BLEU scores
            eval_results = evaluate_model_on_task(
                model=self.lora_model,
                tokenizer=self.tokenizer,
                task=task_name,
                eval_dataset=val_dataset,
                output_dir=self.output_dir,
                device=str(self.device)
            )
            
            results[task_name] = eval_results
            
            # Log results
            bleu_score = eval_results.get('bleu', 0.0)
            num_samples = eval_results.get('num_examples', 0)
            logger.info(f"{task_name} - BLEU: {bleu_score:.4f}, Samples: {num_samples}")
            
            if 'error' in eval_results:
                logger.warning(f"{task_name} evaluation error: {eval_results['error']}")
        
        return results

    def sequential_training_experiment(self):
        """
        Main experiment: Sequential LoRA training on multiple tasks
        
        Args:
            task_sequence: List of task names to train on sequentially
        """
        logger.info("Starting Sequential LoRA Training Experiment")
        logger.info(f"Task sequence: {self.task_sequence}")
        
        # Setup
        self.task_list = self.task_sequence  # Set task list for data loading
        self.setup_base_model()
        self.setup_lora_model()
        self.setup_trainer()  # Create single trainer for all tasks
        
        # Get all task data
        tasks_data_dict = self.get_tasks_data_dict()
        
        # Results tracking
        experiment_results = {
            'task_sequence': self.task_sequence,
            'training_history': [],
            'final_evaluation': {}
        }
        
        # Sequential training loop with single LoRA model
        logger.info("Using single LoRA model for continual learning across all tasks")
        
        for step, task_name in enumerate(self.task_sequence):
            logger.info(f"\n{'='*50}")
            logger.info(f"Step {step + 1}/{len(self.task_sequence)}: Continual training on {task_name}")
            logger.info(f"Same LoRA parameters will be updated with knowledge from {task_name}")
            logger.info(f"{'='*50}")
            
            # Train on current task (continues from previous task's LoRA state)
            self.train_on_task(
                task_name=task_name,
                train_dataset=tasks_data_dict[task_name]['train'],
                val_dataset=tasks_data_dict[task_name]['validation']
            )
            
            
            # Evaluate on all tasks after training on current task
            logger.info(f"\nEvaluating after training on {task_name}...")
            evaluation_results = self.evaluate_on_all_tasks(tasks_data_dict)
            
            # Store results
            step_results = {
                'step': step + 1,
                'trained_task': task_name,
                'evaluation_results': evaluation_results,
                'timestamp': datetime.now().isoformat()
            }
            experiment_results['training_history'].append(step_results)
            
            # Print summary
            self._print_evaluation_summary(task_name, evaluation_results)
            
            # Save intermediate results
            self._save_experiment_results(experiment_results, f"sequential_training_step_{step+1}.json")
            # free_gpu_cache() 
        # Final evaluation
        logger.info(f"\n{'='*50}")
        logger.info("Final Evaluation Complete")
        logger.info(f"{'='*50}")
        
        experiment_results['final_evaluation'] = experiment_results['training_history'][-1]['evaluation_results']
        
        # Save final results
        self._save_experiment_results(experiment_results, "sequential_training_final.json")
        
        # Final cleanup
        
        return experiment_results
        

    def _print_evaluation_summary(self, current_task: str, results: Dict):
        """Print a nice summary of evaluation results"""
        print(f"\n📊 Evaluation Summary after training on {current_task}:")
        print("-" * 70)
        for task, metrics in results.items():
            status = "📈 CURRENT" if task == current_task else "📉"
            bleu_score = metrics.get('bleu', 0.0)
            num_samples = metrics.get('num_examples', 0)
            eval_loss = metrics.get('eval_loss', 'N/A')
            
            if isinstance(eval_loss, float):
                loss_str = f"{eval_loss:.4f}"
            else:
                loss_str = str(eval_loss)
            
            print(f"{status} {task:15} | BLEU: {bleu_score:.4f} | Loss: {loss_str} | Samples: {num_samples}")
            
            if 'error' in metrics:
                print(f"    ⚠️  Error: {metrics['error']}")
        print("-" * 70)

    def _save_experiment_results(self, results: Dict, filename: str):
        """Save experiment results to JSON file"""
        os.makedirs(f"{self.output_dir}/results", exist_ok=True)
        filepath = f"{self.output_dir}/results/{filename}"
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":

    
    # Initialize training pipeline
    trainer = LoRATrainingPipeline(
        model_name="Salesforce/codet5p-220m",
        output_dir=output_dir,
        input_dir=input_dir,
        task_sequence=["BFP_java", "CodeTrans_java_to_csharp"]
    )
    

    
    # Run sequential training experiment
    results = trainer.sequential_training_experiment()
    
    print("\nSequential LoRA Training Experiment Completed!")
    print(f"Results saved in: {trainer.output_dir}/results/")
    