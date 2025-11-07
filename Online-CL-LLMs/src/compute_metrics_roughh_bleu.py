import string
import json
import os
import argparse
import logging
import collections
import math

from rouge import rouge_scorer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
GPT2TOKENIZER = os.path.join(CURRENT_DIR, "../data/gpt2tokenizer")


# class GPTTokenizer:
#     gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2TOKENIZER, max_length=1e5)

#     def tokenize(self, s):
#         tokens = self.gpt_tokenizer.tokenize(s)
#         # GPT2 uses Byte-level BPE, which will include space as part of the word. 
#         # But for the first word of a sentence, there is no space before it. 
#         # So, we remove all the added spaces ("Ġ"). 
#         tokens = [t.lstrip("Ġ") for t in tokens]
#         return tokens


# xlingual_tokenizer = GPTTokenizer()


class BleuScorer:
    """BLEU score computation class."""
    
    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams up to a given max_order from a token list."""
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu(self, reference_corpus, translation_corpus, max_order=1, smooth=False):
        """
        Computes BLEU score of translated segments against one or more references.

        reference_corpus: list of lists of references for each translation.
                        Each reference should be a tokenized list.
        translation_corpus: list of tokenized translations to score.
        """
        # Handle empty corpora
        if not reference_corpus or not translation_corpus:
            return 0.0
        
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0

        for (references, translation) in zip(reference_corpus, translation_corpus):
            # Handle empty references or translations
            if not references or not translation or not any(references):
                continue
                
            # references is a list of token lists; translation is a single token list
            reference_length += min(len(r) for r in references if r)
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
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                    possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        # Handle zero reference length
        if reference_length == 0:
            return 0.0

        ratio = float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1.0
        else:
            if reference_length == 0:
                bp = 0.0
            else:
                bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp
        return bleu  # typically a float in [0..1]


bleu_scorer = BleuScorer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def bleu_score(prediction, ground_truth, xlingual=False):
    """Compute BLEU score between prediction and ground truth."""
    # Handle empty strings
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    
    if xlingual:
        tokenizer = xlingual_tokenizer
        pred_tokens = tokenizer.tokenize(prediction)
        ref_tokens = tokenizer.tokenize(ground_truth)
    else:
        # Simple whitespace tokenization for default case
        pred_tokens = prediction.split()
        ref_tokens = ground_truth.split()
        print("Pred tokens:", pred_tokens)
        print("Ref tokens:", ref_tokens)
    # Handle empty token lists
    if not pred_tokens or not ref_tokens:
        print("Empty tokens after tokenization.")
        return 0.0
    
    # BLEU expects reference_corpus as list of lists and translation_corpus as list
    reference_corpus = [[ref_tokens]]  # Single reference wrapped in list
    translation_corpus = [pred_tokens]
    
    return bleu_scorer.compute_bleu(reference_corpus, translation_corpus)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL, bleu = 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        bleu += metric_max_over_ground_truths(
            bleu_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    bleu = 100.0 * bleu / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "eval_rougeL": rougeL, "bleu": bleu}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results