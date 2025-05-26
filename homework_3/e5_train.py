import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        similarity = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature

        labels = torch.arange(similarity.size(0), device=similarity.device)

        loss = F.cross_entropy(similarity, labels)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

        pos_dist = torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1)
        neg_dist = torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

class RetrievalDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        mode: str = "contrastive",
        model: AutoModel = None,
        device: str = "cuda"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model = model
        self.device = device

        # Pre-compute document embeddings for hard negative mining
        if self.model is not None:
            self.doc_embeddings = self._compute_document_embeddings()
            self.similarity_matrix = cosine_similarity(self.doc_embeddings)
        
    def _compute_document_embeddings(self) -> np.ndarray:
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for item in tqdm(self.data, desc="Computing document embeddings"):
                encoding = self.tokenizer(
                    item['passage'],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                output = self.model(**encoding)
                embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
                
        return np.array(embeddings)
    
    def _get_hard_negative(self, idx: int) -> int:
        similarities = self.similarity_matrix[idx]

        sorted_indices = np.argsort(similarities)[::-1]

        for neg_idx in sorted_indices[1:]:  # Skip the first one as it's the same document
            if neg_idx != idx:
                return neg_idx                

        return random.randint(0, len(self.data) - 1)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query = item['question']
        positive_doc = item['passage']
        
        if self.model is not None:
            negative_idx = self._get_hard_negative(idx)
        else:
            negative_idx = random.randint(0, len(self.data) - 1)
            while negative_idx == idx:
                negative_idx = random.randint(0, len(self.data) - 1)
                
        negative_doc = self.data[negative_idx]['passage']
        
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive_doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            negative_doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.mode == "contrastive":
            return {
                'query_input_ids': query_encoding['input_ids'].squeeze(),
                'query_attention_mask': query_encoding['attention_mask'].squeeze(),
                'positive_input_ids': positive_encoding['input_ids'].squeeze(),
                'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
                'negative_input_ids': negative_encoding['input_ids'].squeeze(),
                'negative_attention_mask': negative_encoding['attention_mask'].squeeze()
            }
        else:
            return {
                'anchor_input_ids': query_encoding['input_ids'].squeeze(),
                'anchor_attention_mask': query_encoding['attention_mask'].squeeze(),
                'positive_input_ids': positive_encoding['input_ids'].squeeze(),
                'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
                'negative_input_ids': negative_encoding['input_ids'].squeeze(),
                'negative_attention_mask': negative_encoding['attention_mask'].squeeze()
            }

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

class E5Trainer:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cuda",
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 3,
        max_length: int = 512
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        self.contrastive_loss = ContrastiveLoss().to(self.device)
        self.triplet_loss = TripletLoss().to(self.device)
        
    def train(
        self,
        train_data: List[Dict[str, Any]],
        output_dir: str,
        mode: str = "contrastive"
    ):
        train_dataset = RetrievalDataset(
            train_data,
            self.tokenizer,
            self.max_length,
            mode,
            self.model,
            self.device
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_loader) * self.num_epochs
        )
        
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if mode == "contrastive":
                    query_embeddings = self.model(
                        input_ids=batch['query_input_ids'],
                        attention_mask=batch['query_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    positive_embeddings = self.model(
                        input_ids=batch['positive_input_ids'],
                        attention_mask=batch['positive_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    negative_embeddings = self.model(
                        input_ids=batch['negative_input_ids'],
                        attention_mask=batch['negative_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    loss = self.contrastive_loss(query_embeddings, positive_embeddings)
                    
                else:
                    anchor_embeddings = self.model(
                        input_ids=batch['anchor_input_ids'],
                        attention_mask=batch['anchor_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    positive_embeddings = self.model(
                        input_ids=batch['positive_input_ids'],
                        attention_mask=batch['positive_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    negative_embeddings = self.model(
                        input_ids=batch['negative_input_ids'],
                        attention_mask=batch['negative_attention_mask']
                    ).last_hidden_state[:, 0, :]
                    
                    loss = self.triplet_loss(
                        anchor_embeddings,
                        positive_embeddings,
                        negative_embeddings
                    )
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    self.model.state_dict(),
                    Path(output_dir) / f"best_model_{mode}.pt"
                )
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        
        test_queries = [item['question'] for item in test_data]
        test_documents = [item['passage'] for item in test_data]
        
        query_embeddings = []
        doc_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_queries), self.batch_size), desc="Encoding queries"):
                batch_queries = test_queries[i:i + self.batch_size]
                encoded = self.tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                query_embeddings.append(batch_embeddings.cpu().numpy())
            
            for i in tqdm(range(0, len(test_documents), self.batch_size), desc="Encoding documents"):
                batch_docs = test_documents[i:i + self.batch_size]
                encoded = self.tokenizer(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                doc_embeddings.append(batch_embeddings.cpu().numpy())
        
        query_embeddings = np.vstack(query_embeddings)
        doc_embeddings = np.vstack(doc_embeddings)
        
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
        
        return {
            'recall@1': recall_1,
            'recall@3': recall_3,
            'recall@10': recall_10,
            'mrr': mrr
        }

def main():
    train_data = load_data("data/train.json")
    test_data = load_data("data/test.json")
    
    
    trainer = E5Trainer(
        model_name="intfloat/multilingual-e5-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=2e-5,
        batch_size=32,
        num_epochs=3,
        max_length=512
    )

    print("Starting training with hard negatives...")
    trainer.train(
        train_data=train_data,
        output_dir="models/e5_hard_negatives",
        mode="triplet"
    )
    
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_data)
    
    print("\nTest Results:")
    print(f"Recall@1: {metrics['recall@1']:.4f}")
    print(f"Recall@3: {metrics['recall@3']:.4f}")
    print(f"Recall@10: {metrics['recall@10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")

    results = {
        'recall@1': metrics['recall@1'],
        'recall@3': metrics['recall@3'],
        'recall@10': metrics['recall@10'],
        'mrr': metrics['mrr']
    }

    with open("results/e5_hard_negatives_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 