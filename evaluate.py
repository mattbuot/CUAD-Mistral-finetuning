import json
import re
import string


def evaluate_predictions(file_name: str):
    """Evaluate predictions against labels from a JSONL file."""

    predictions, labels = read_predictions_and_labels(file_name=file_name)

    try:
        results = compute_precision_recall(predictions, labels)
        print(f"Evaluation results: {results}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


def compute_precision_recall(predictions: list[list[str]], labels: list[list[str]]):
    assert len(predictions) == len(labels), "Predictions and labels must have the same length."

    precision = 0
    recall = 0

    for prediction, label in zip(predictions, labels):
        normalized_predictions = [normalize_answer(p) for p in prediction]
        normalized_labels = [normalize_answer(l) for l in label]

        precision += precision_score(normalized_predictions, normalized_labels)
        recall += recall_score(normalized_predictions, normalized_labels)

    total = len(predictions)
    return {
        "precision": precision / total,
        "recall": recall / total
    }


def precision_score(prediction: list[str], ground_truth: list[str]) -> float:
    """Calculate precision score"""

    if len(prediction) == 0:
        return 0.0

    precision = sum(metric_max_over_ground_truths(exact_match_score, p, ground_truth) for p in prediction) / len(prediction)

    return precision


def recall_score(prediction, ground_truth):
    """Calculate recall score"""

    if len(ground_truth) == 0:
        return 0.0
    
    recall = sum(metric_max_over_ground_truths(exact_match_score, p, ground_truth) for p in prediction) / len(ground_truth)

    return recall


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate max metric over all ground truths"""
    if not ground_truths:
        return 0.0
    scores = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores.append(score)
    return max(scores)


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    return int(prediction == ground_truth)


def normalize_answer(s: str):
    """Normalize answer for comparison (SQuAD-style)
    
    # copied from https://github.com/anishdulal/llm-cuad-eval/blob/main/finetune_evaluate_colab.ipynb
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def read_predictions_and_labels(file_name: str) -> tuple[list[list[str]], list[list[str]]]:
    """Read predictions and labels from a JSONL file."""
    predictions = []
    labels = []

    with open(file_name, "r") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["prediction"])
            labels.append(data["label"])

    return predictions, labels


if __name__ == "__main__":

    file_name = "data/cuad_test_predictions.jsonl"
    evaluate_predictions(file_name=file_name)
