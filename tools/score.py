from collections import Counter
import math
from tqdm import tqdm

def modified_precision(reference, candidate, n):
    counts = Counter()
    max_counts = Counter()

    for i in range(len(candidate) - n + 1):
        n_gram = tuple(candidate[i:i+n])
        counts[n_gram] += 1

    for ref in reference:
        ref_counts = Counter()
        for i in range(len(ref) - n + 1):
            n_gram = tuple(ref[i:i+n])
            ref_counts[n_gram] += 1
        for n_gram in counts:
            max_counts[n_gram] = max(max_counts[n_gram], ref_counts[n_gram])

    clipped_counts = {n_gram: min(counts[n_gram], max_counts[n_gram]) for n_gram in counts}
    numerator = sum(clipped_counts.values())
    denominator = max(1, len(candidate) - n + 1)
    return 0 if denominator == 0 else numerator / denominator

def bleu(reference, candidate, max_n=4):
    weights = [1 / max_n] * max_n
    precisions = [modified_precision(reference, candidate, i) for i in range(1, max_n + 1)]
    # Avoid division by zero
    log_precision_sum = sum(w * math.log(p) if p != 0 else 0 for w, p in zip(weights, precisions))
    bleu_score = math.exp(log_precision_sum) if all(precisions) else 0
    return bleu_score

def avg_window_bleu(references, candidates, n = 4):
    r_sum = 0
    for reference in tqdm(references, desc="References", position=0):
        c_sum = 0
        for candidate in tqdm(candidates, desc="Candidates", position=1, leave=False):
            if len(candidate) < len(reference):
                score = bleu([reference], candidate)
                c_sum += score
                continue
            score = 0
            length = len(candidate) - len(reference) + 1
            for i in tqdm(range(length), desc="Window", position=2, leave=False):
                score_window = bleu([reference], candidate[i:i+len(reference)])
                score += score_window
            score /= length
            c_sum += score
        c_avg = c_sum / len(candidates)
        r_sum += c_avg
    return r_sum / len(references)

def deg_calulate(output_texts, w_output_texts, targets):
    sum = 0
    w_sum = 0 
    for i in range(len(output_texts)):
        score = bleu(targets[i], output_texts[i])
        w_score = bleu(targets[i], w_output_texts[i])
        sum += score
        w_sum += w_score
    return (w_sum - sum) / len(output_texts)