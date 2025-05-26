import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

class NeuralRetriever(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        device: str = "cuda"
    ):
        """
        Initialize neural retriever.
        
        Args:
            model_name: Name of the transformer model to use
            max_length: Maximum sequence length
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.index = None
        self.documents = []
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**encoded)
                # Use [CLS] token embedding as sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(embeddings)
    
    def build_index(self, documents: List[str], use_faiss: bool = True) -> None:
        """
        Build search index from documents.
        
        Args:
            documents: List of document texts
            use_faiss: Whether to use FAISS for efficient search
        """
        self.documents = documents
        embeddings = self.encode(documents)
        
        if use_faiss:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index = embeddings
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing document text and similarity score
        """
        if self.index is None:
            raise ValueError("Index must be built before retrieval")
            
        # Encode query
        query_embedding = self.encode([query])
        
        if isinstance(self.index, faiss.Index):
            # FAISS search
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Cosine similarity with numpy
            scores = np.dot(query_embedding, self.index.T)[0]
            indices = np.argsort(scores)[-top_k:][::-1]
            scores = scores[indices]
        
        # Prepare results
        results = []
        for score, idx in zip(scores, indices):
            if score >= similarity_threshold:
                results.append({
                    'text': self.documents[idx],
                    'score': float(score)
                })
                
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve top-k most similar documents for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of lists containing retrieval results for each query
        """
        return [self.retrieve(query, top_k, similarity_threshold) for query in queries] 