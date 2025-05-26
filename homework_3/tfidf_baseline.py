import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

def calculate_recall_at_k(predictions: List[List[int]], ground_truth: List[int], k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        predictions: List of predicted document indices for each query
        ground_truth: List of ground truth document indices
        k: Number of top predictions to consider
        
    Returns:
        float: Recall@K score
    """
    correct = 0
    total = len(ground_truth)

    for pred, true in zip(predictions, ground_truth):
        if true in pred[:k]:
            correct += 1

    return correct / total if total > 0 else 0.0

def calculate_mrr(predictions: List[List[int]], ground_truth: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List of predicted document indices for each query
        ground_truth: List of ground truth document indices
        
    Returns:
        float: MRR score
    """
    reciprocal_ranks = []

    for pred, true in zip(predictions, ground_truth):
        try:
            rank = pred.index(true) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
            
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    train_data = load_data("data/train.json")
    test_data = load_data("data/test.json")

    train_documents = [item['passage'] for item in train_data]
    test_queries = [item['question'] for item in test_data]
    test_documents = [item['passage'] for item in test_data]

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )

    vectorizer.fit(train_documents)

    test_doc_vectors = vectorizer.transform(test_documents)
    test_query_vectors = vectorizer.transform(test_queries)
    
    similarities = cosine_similarity(test_query_vectors, test_doc_vectors)

    predictions = []
    for sim_row in similarities:
        sorted_indices = np.argsort(sim_row)[::-1]
        predictions.append(sorted_indices.tolist())
    
    ground_truth = list(range(len(test_data)))
    
    recall_1 = calculate_recall_at_k(predictions, ground_truth, k=1)
    recall_3 = calculate_recall_at_k(predictions, ground_truth, k=3)
    recall_10 = calculate_recall_at_k(predictions, ground_truth, k=10)
    mrr = calculate_mrr(predictions, ground_truth)
    
    print("\nTF-IDF Baseline Results:")
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    results = {
        'recall@1': recall_1,
        'recall@3': recall_3,
        'recall@10': recall_10,
        'mrr': mrr
    }
    
    with open("results/tfidf_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 