import os
import torch
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Any
import collections
import math
from torch.utils.data import DataLoader
import logging
import time
import json
from datetime import datetime
from transformers import DataCollatorForSeq2Seq



class CustomBLEU:
    """Custom BLEU implementation"""
    
    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams up to a given max_order from a token list."""
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu(self, reference_corpus, translation_corpus, max_order=4, smooth=False):
        """
        Computes BLEU score of translated segments against one or more references.

        reference_corpus: list of lists of references for each translation.
                          Each reference should be a tokenized list.
        translation_corpus: list of tokenized translations to score.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0

        for (references, translation) in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)

            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts

            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]

            for order in range(1, max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                 (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length if reference_length > 0 else 0
        if ratio > 1.0:
            bp = 1.0
        else:
            bp = math.exp(1 - 1. / ratio) if ratio > 0 else 0

        bleu = geo_mean * bp
        return bleu  # float in [0,1]


# Initialize custom BLEU scorer
bleu_scorer = CustomBLEU()


def evaluate_model_on_task(model, 
                           tokenizer,
                           task: str, 
                           eval_dataset, 
                           device: str = "cuda",
                           log_samples: bool = True,
                           output_dir: str = None,
                           batch_size: int = 4) -> Dict[str, Any]:

    logger = logging.getLogger(__name__)
    
    if len(eval_dataset) == 0:
        return {
            'task': task,
            'num_examples': 0,
            'bleu': 0.0,
            'error': 'No evaluation data available'
        }
    
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",   # pad dynamically to max length in batch
        max_length=512, 
        # truncation=True, 
        return_tensors="pt"
    )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    model.eval()
    all_predictions, all_references = [], []
    total_loss, total_samples = 0.0, 0
    start_time = time.time()
    evaluation_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                total_loss += outputs.loss.item() * input_ids.shape[0]
                
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                for i in range(input_ids.shape[0]):
                    pred_tokens = generated_ids[i]
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    
                    ref_tokens = [l for l in labels[i].cpu().numpy() if l != -100]
                    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                    
                    all_predictions.append(pred_text)
                    all_references.append([ref_text])
                    
                    if log_samples and batch_idx <=10:
                        input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                        ref_words = set(ref_text.lower().split())
                        pred_words = set(pred_text.lower().split())
                        overlap = len(ref_words & pred_words) / len(ref_words) if ref_words else 0.0
                        
                        sample_data = {
                            'task': task,
                            'input': input_text,
                            'reference': ref_text,
                            'prediction': pred_text,
                            'word_overlap': overlap,
                            'reference_length': len(ref_text.split()),
                            'prediction_length': len(pred_text.split())
                        }
                        evaluation_samples.append(sample_data)
                        
                        logger.info(f"\n{task} - Sample {i+1}:")
                        logger.info(f"Input: {input_text}")
                        logger.info(f"Reference: {ref_text}")
                        logger.info(f"Prediction: {pred_text}")
                        logger.info(f"Word Overlap: {overlap:.2%}")
                
                total_samples += input_ids.shape[0]
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    try:
        pred_tokens = [pred.split() for pred in all_predictions]
        ref_tokens_list = [[ref.split()] for [ref] in all_references]
        
        bleu_score = bleu_scorer.compute_bleu(
            reference_corpus=ref_tokens_list,
            translation_corpus=pred_tokens,
            max_order=4,
            smooth=True
        ) * 100  # scaled to 0-100
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        bleu_score = 0.0
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    eval_time = time.time() - start_time
    samples_per_second = total_samples / eval_time if eval_time > 0 else 0.0
    
    if log_samples and evaluation_samples and output_dir:
        try:
            save_dir = os.path.join(output_dir, "prediction")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"eval_{task}_{timestamp}.jsonl")
            
            with open(filename, 'w', encoding='utf-8') as f:
                for sample in evaluation_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Saved {len(evaluation_samples)} evaluation samples to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save evaluation samples: {e}")
    
    return {
        'task': task,
        'num_examples': total_samples,
        'bleu': bleu_score,
        'eval_loss': avg_loss,
        'eval_runtime': eval_time,
        'eval_samples_per_second': samples_per_second
    }
