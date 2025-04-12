import torch
from torch import nn, Tensor
from typing import Dict, List

class ResidualBlockWithDropout(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x

class RegularizedModel(nn.Module):
    HOME_OWNERSHIP_SIZE = 4
    LOAN_INTENT_SIZE = 6  
    LOAN_GRADE_SIZE = 7
    DEFAULT_HISTORY_SIZE = 2
    NUM_NUMERIC_FEATURES = 7

    def __init__(self, hidden_size: int, drop_p: float):
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            'home': nn.Embedding(self.HOME_OWNERSHIP_SIZE, hidden_size),
            'intent': nn.Embedding(self.LOAN_INTENT_SIZE, hidden_size),
            'grade': nn.Embedding(self.LOAN_GRADE_SIZE, hidden_size),
            'default_history': nn.Embedding(self.DEFAULT_HISTORY_SIZE, hidden_size)
        })
        self.embedding_dropout = nn.Dropout(drop_p)

        self.numeric_bn = nn.BatchNorm1d(self.NUM_NUMERIC_FEATURES)
        self.numeric_dropout = nn.Dropout(drop_p)
        
        self.numeric_projection = nn.Linear(self.NUM_NUMERIC_FEATURES, hidden_size)
        
        self.features_bn = nn.BatchNorm1d(hidden_size)
        self.features_dropout = nn.Dropout(drop_p)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlockWithDropout(hidden_size, drop_p) for _ in range(3)
        ])
        
        self.final_bn = nn.BatchNorm1d(hidden_size)
        self.final_dropout = nn.Dropout(drop_p)
        
        self.classifier = nn.Linear(hidden_size, 1)

    def _process_numeric_features(self, numeric_features: Dict[str, Tensor]) -> Tensor:
        numeric = torch.stack([
            (numeric_features['person_age'] - 20) / 100,
            (numeric_features['person_income'] - 4e3) / 1.2e6,
            numeric_features['person_emp_length'] / 123,
            (numeric_features['loan_amnt'] - 500) / 35000,
            (numeric_features['loan_int_rate'] - 5.4) / 23.3,
            numeric_features['loan_percent_income'],
            (numeric_features['cb_person_cred_hist_length'] - 2) / 28
        ], dim=-1)
        
        numeric = self.numeric_bn(numeric)
        numeric = self.numeric_dropout(numeric)
        return self.numeric_projection(numeric)

    def forward(self, cat_features: Dict[str, Tensor], numeric_features: Dict[str, Tensor]) -> Tensor:
        embeddings = [
            self.embeddings['home'](cat_features['person_home_ownership']),
            self.embeddings['intent'](cat_features['loan_intent']),
            self.embeddings['grade'](cat_features['loan_grade']),
            self.embeddings['default_history'](cat_features['cb_person_default_on_file'])
        ]
        embeddings = [self.embedding_dropout(emb) for emb in embeddings]
        
        numeric = self._process_numeric_features(numeric_features)
        
        x = sum(embeddings) + numeric
        x = self.features_bn(x)
        x = self.features_dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.final_bn(x)
        x = self.final_dropout(x)
        
        return self.classifier(x).squeeze(-1)