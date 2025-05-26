from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever:
    def __init__(self, max_features: int = 50000, ngram_range: tuple = (1, 2), min_df: int = 2):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency for terms
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english'
        )
        self.documents = []
        self.document_vectors = None
        
    def fit(self, documents: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on the documents.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents)
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document text and similarity score
        """
        if self.document_vectors is None:
            raise ValueError("Model must be fitted before retrieval")
            
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': float(similarities[idx])
            })
            
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve top-k most similar documents for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of lists containing retrieval results for each query
        """
        return [self.retrieve(query, top_k) for query in queries] 