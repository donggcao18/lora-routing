

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType
import logging
from datetime import datetime

# Import the task prototype extractor
from architecture import TaskPrototypeExtractor, TaskPrototype

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRABlock:
    block_id: str
    model: Any  # PEFT model with LoRA adapter
    key_vector: np.ndarray  # Representative embedding for similarity checking
    task_prototypes: List[TaskPrototype]  # All tasks assigned to this block
    usage_count: int = 0

class LoRARouter:
    
    def __init__(self,
                 base_model,
                 prototype_extractor: TaskPrototypeExtractor,
                 similarity_threshold: float = 0.7,
                 lora_config: Optional[LoraConfig] = None):

        self.base_model = base_model
        self.prototype_extractor = prototype_extractor
        self.similarity_threshold = similarity_threshold        
        # LoRA configuration
        if lora_config is None:
            self.lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # T5 specific
            )
        else:
            self.lora_config = lora_config
        
        # LoRA blocks for task routing
        self.lora_blocks: Dict[str, LoRABlock] = {}
        self.next_block_id = 0
        
        # Routing statistics
        self.routing_history = []
        self.total_routes = 0
        self.reuse_count = 0
        self.new_block_count = 0
        
        logger.info(f"LoRA Router initialized with threshold={similarity_threshold}")
        logger.info("Sequential task routing: Each task will be routed to best matching block or create new block")
    
    def _generate_block_id(self) -> str:
        """Generate unique block ID"""
        block_id = f"lora_block_{self.next_block_id}"
        self.next_block_id += 1
        return block_id
    
    def _create_lora_block(self, task_prototype: TaskPrototype) -> LoRABlock:
        
        block_id = self._generate_block_id()
        
        peft_model = get_peft_model(self.base_model, self.lora_config)        
        key_vector = task_prototype.prototype_embedding.copy()
        lora_block = LoRABlock(
            block_id=block_id,
            model=peft_model,
            key_vector=key_vector,
            task_prototypes=[task_prototype],
            usage_count=1
        )
        
        # Add to blocks
        self.lora_blocks[block_id] = lora_block
        self.new_block_count += 1
        
        logger.info(f"Created new LoRA block {block_id} for task: {task_prototype.task_id}")
        return lora_block
    
    
    def _find_best_matching_block(self, task_prototype: TaskPrototype) -> Tuple[Optional[str], float]:
        """Find block with highest similarity to task prototype using key vectors"""
        if not self.lora_blocks:
            return None, 0.0
        
        best_block_id = None
        best_similarity = 0.0
        
        task_embedding = task_prototype.prototype_embedding
        
        for block_id, lora_block in self.lora_blocks.items():
            # Compute cosine similarity between task embedding and block's key vector
            similarity = np.dot(task_embedding, lora_block.key_vector) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(lora_block.key_vector)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_block_id = block_id
        
        return best_block_id, best_similarity
    
    def _update_block_with_task(self, block: LoRABlock, task_prototype: TaskPrototype):
        """Update existing block to handle new task - update key vector and add task"""
        # Add new task prototype to block
        block.task_prototypes.append(task_prototype)
        
        # Update key vector as weighted average of all task embeddings
        all_embeddings = np.array([tp.prototype_embedding for tp in block.task_prototypes])
        block.key_vector = np.mean(all_embeddings, axis=0)
        
        # Update usage
        block.usage_count += 1
        
        logger.info(f"Updated LoRA block {block.block_id} with new task: {task_prototype.task_id}. "
                   f"Now handles {len(block.task_prototypes)} tasks.")
    
    def route_task(self,
                   task_id: str,
                   sample_data: List[Dict],
                   language: str) -> Tuple[str, str, float, TaskPrototype]:

        logger.info(f"Routing task {task_id} for language {language}")
        
        # Extract task prototype
        task_prototype = self.prototype_extractor.extract_prototype(
            task_id=task_id,
            sample_data=sample_data,
            language=language
        )
        
        # Find best matching block
        best_block_id, best_similarity = self._find_best_matching_block(task_prototype)
        
        routing_decision = None
        selected_block_id = None
        
        if best_block_id is not None and best_similarity >= self.similarity_threshold:
            # Reuse existing block and update it with new task
            selected_block_id = best_block_id
            routing_decision = 'reuse'
            
            # Update block with new task
            self._update_block_with_task(self.lora_blocks[best_block_id], task_prototype)
            self.reuse_count += 1
            
            logger.info(f"Reusing block {best_block_id} (similarity: {best_similarity:.3f})")
            
        else:
            # Create new block
            new_block = self._create_lora_block(task_prototype)
            selected_block_id = new_block.block_id
            routing_decision = 'new'
            
            # Update usage
            new_block.usage_count = 1
            
            logger.info(f"Created new block {selected_block_id} (best similarity: {best_similarity:.3f})")
        
        # Record routing history
        self.total_routes += 1
        routing_record = {
            'task_id': task_id,
            'block_id': selected_block_id,
            'decision': routing_decision,
            'similarity': best_similarity,
            'language': language,
            'timestamp': datetime.now().isoformat()
        }
        self.routing_history.append(routing_record)
        
        return selected_block_id, routing_decision, best_similarity, task_prototype
    
    def get_block_model(self, block_id: str):
        """Get the PEFT model for a specific block"""
        if block_id not in self.lora_blocks:
            raise ValueError(f"Block {block_id} not found")
        return self.lora_blocks[block_id].model
    
    def get_block_info(self, block_id: str) -> Dict:
        """Get detailed information about a specific block"""
        if block_id not in self.lora_blocks:
            raise ValueError(f"Block {block_id} not found")
        
        block = self.lora_blocks[block_id]
        return {
            'block_id': block.block_id,
            'languages': list(set(tp.language for tp in block.task_prototypes)),
            'tasks': [tp.task_id for tp in block.task_prototypes],
            'task_count': len(block.task_prototypes),
            'usage_count': block.usage_count,
            'key_vector_shape': block.key_vector.shape if block.key_vector is not None else None
        }
    
    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing statistics"""
        if self.total_routes == 0:
            reuse_rate = 0.0
        else:
            reuse_rate = self.reuse_count / self.total_routes
        
        # Language distribution
        language_stats = {}
        for block in self.lora_blocks.values():
            # Get all languages handled by this block
            languages = set(tp.language for tp in block.task_prototypes)
            for lang in languages:
                if lang not in language_stats:
                    language_stats[lang] = {'blocks': 0, 'usage': 0}
                language_stats[lang]['blocks'] += 1
                language_stats[lang]['usage'] += block.usage_count
        
        # Block efficiency
        block_stats = {}
        for block_id, block in self.lora_blocks.items():
            # Get all languages and tasks handled by this block
            languages = list(set(tp.language for tp in block.task_prototypes))
            tasks = [tp.task_id for tp in block.task_prototypes]
            
            block_stats[block_id] = {
                'languages': languages,
                'tasks': tasks,
                'task_count': len(block.task_prototypes),
                'usage_count': block.usage_count,
            }
        
        return {
            'total_routes': self.total_routes,
            'reuse_count': self.reuse_count,
            'new_blocks_created': self.new_block_count,
            'reuse_rate': reuse_rate,
            'active_blocks': len(self.lora_blocks),
            'similarity_threshold': self.similarity_threshold,
            'language_stats': language_stats,
            'block_stats': block_stats
        }
    
    def print_routing_summary(self):
        """Print comprehensive routing summary"""
        stats = self.get_routing_statistics()
        
        print(f"\n{'='*60}")
        print("LORA ROUTER STATISTICS")
        print(f"{'='*60}")
        
        print(f"Total routes: {stats['total_routes']}")
        print(f"Blocks reused: {stats['reuse_count']}")
        print(f"New blocks created: {stats['new_blocks_created']}")
        print(f"Reuse rate: {stats['reuse_rate']:.1%}")
        print(f"Active blocks: {stats['active_blocks']}")
        print(f"Similarity threshold: {stats['similarity_threshold']}")
        
        print(f"\nLanguage Distribution:")
        for lang, lang_stats in stats['language_stats'].items():
            print(f"  {lang}: {lang_stats['blocks']} blocks, {lang_stats['usage']} total usage")
        
        print(f"\nBlock Details:")
        for block_id, block_stats in stats['block_stats'].items():
            languages_str = ", ".join(block_stats['languages'])
            print(f"  {block_id}: {languages_str}, "
                  f"{block_stats['task_count']} tasks, "
                  f"used {block_stats['usage_count']} times")
        
        if stats['total_routes'] > 0:
            print(f"\nEfficiency Metrics:")
            avg_usage = sum(b['usage_count'] for b in stats['block_stats'].values()) / len(stats['block_stats']) if stats['block_stats'] else 0
            print(f"  Average usage per block: {avg_usage:.1f}")
            memory_efficiency = (stats['new_blocks_created'] / stats['total_routes']) * 100
            print(f"  Memory efficiency: {100 - memory_efficiency:.1f}% (lower is better)")
        
        print(f"{'='*60}")
    
    def save_routing_history(self, filepath: str):
        """Save routing history to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump({
                'routing_history': self.routing_history,
                'statistics': self.get_routing_statistics()
            }, f, indent=2)
        
        logger.info(f"Routing history saved to {filepath}")
    
    def reset_statistics(self):
        """Reset routing statistics"""
        self.routing_history = []
        self.total_routes = 0
        self.reuse_count = 0
        self.new_block_count = 0
        
        # Reset block usage counts
        for block in self.lora_blocks.values():
            block.usage_count = 0
        
        logger.info("Routing statistics reset")


# Example usage and testing
if __name__ == "__main__":
    print("🔄 LoRA Router Example Usage")
    print("="*50)
    
    # This is just a demonstration - in practice you'd use real models
    class MockModel:
        """Mock model for testing"""
        def parameters(self):
            return []
    
    try:
        # Initialize components
        mock_model = MockModel()
        prototype_extractor = TaskPrototypeExtractor()
        
        router = LoRARouter(
            base_model=mock_model,
            prototype_extractor=prototype_extractor,
            similarity_threshold=0.6,
        )
        
        # Sample tasks
        python_math_samples = [
            {"input": "[LANGUAGE] python [COMMAND] Add two numbers", 
             "output": "def add(a, b):\n    return a + b"}
        ] * 10
        
        python_string_samples = [
            {"input": "[LANGUAGE] python [COMMAND] Reverse a string", 
             "output": "def reverse(s):\n    return s[::-1]"}
        ] * 10
        
        java_samples = [
            {"input": "[LANGUAGE] java [COMMAND] Add two integers", 
             "output": "public int add(int a, int b) {\n    return a + b;\n}"}
        ] * 10
        
        # Route tasks
        tasks = [
            ("python_math_1", python_math_samples, "python"),
            ("python_string_1", python_string_samples, "python"),
            ("java_math_1", java_samples, "java"),
            ("python_math_2", python_math_samples, "python"),  # Should reuse
            ("python_string_2", python_string_samples, "python"),  # Should reuse
        ]
        
        print("Routing tasks...")
        for task_id, samples, language in tasks:
            try:
                block_id, decision, similarity, prototype = router.route_task(
                    task_id, samples, language
                )
                print(f"✅ Task {task_id}: Block {block_id}, Decision: {decision}, "
                      f"Similarity: {similarity:.3f}")
            except Exception as e:
                print(f"❌ Error routing {task_id}: {e}")
        
        # Print summary
        router.print_routing_summary()
        
    except Exception as e:
        print(f"❌ Error in example: {e}")
        print("Note: This example requires proper model setup for full functionality")
