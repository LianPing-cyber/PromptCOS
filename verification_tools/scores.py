from bert_score import score as bert_score_fn

def calculate_similarity(content, mark):
    max_similarity = 0
    mark_len = len(mark)
    if len(content) < mark_len:
        return 0
    for i in range(len(content) - mark_len + 1):
        substring = content[i:i + mark_len]
        matching_chars = sum(1 for j in range(mark_len) if substring[j] == mark[j])
        similarity = matching_chars / mark_len
        max_similarity = max(max_similarity, similarity)
    return max_similarity

def score_ws(contents, marks):
    similarities = []
    for content, mark in zip(contents, marks):
        similarity = calculate_similarity(content, mark)
        similarities.append(similarity)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    max_similarity = max(similarities) if similarities else 0
    min_similarity = min(similarities) if similarities else 0
    return avg_similarity, max_similarity, min_similarity

def score_acc(classer, answers, references):
    correct = 0
    total = len(answers)
    for answer, reference in zip(answers, references):
        if classer.judge(answer, reference):
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

def score_bert_distance(texts1, texts2):
    _, _, F1_1 = bert_score_fn(texts1, texts2, lang='en', verbose=False)
    return F1_1.mean().item()