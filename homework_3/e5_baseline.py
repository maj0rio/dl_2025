import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from tqdm import tqdm

def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_recall_at_k(predictions: List[List[int]], ground_truth: List[int], k: int) -> float:
    correct = 0
    total = len(ground_truth)
    
    for pred, true in zip(predictions, ground_truth):
        if true in pred[:k]:
            correct += 1
            
    return correct / total if total > 0 else 0.0

def calculate_mrr(predictions: List[List[int]], ground_truth: List[int]) -> float:
    reciprocal_ranks = []
    
    for pred, true in zip(predictions, ground_truth):
        try:
            rank = pred.index(true) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
            
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

class E5Retriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(embeddings)

def main():
    test_data = load_data("data/test.json")
    
    test_queries = [item['question'] for item in test_data]
    test_documents = [item['passage'] for item in test_data]
    
    retriever = E5Retriever()

    print("Encoding queries...")
    query_embeddings = retriever.encode(test_queries)
    print("Encoding documents...")
    doc_embeddings = retriever.encode(test_documents)
    
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    
    predictions = []
    for sim_row in similarities:
        sorted_indices = np.argsort(sim_row)[::-1]
        predictions.append(sorted_indices.tolist())
    
    ground_truth = list(range(len(test_data)))
    
    recall_1 = calculate_recall_at_k(predictions, ground_truth, k=1)
    recall_3 = calculate_recall_at_k(predictions, ground_truth, k=3)
    recall_10 = calculate_recall_at_k(predictions, ground_truth, k=10)
    mrr = calculate_mrr(predictions, ground_truth)
    
    print("\nE5 Baseline Results:")
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
    
    with open("results/e5_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 