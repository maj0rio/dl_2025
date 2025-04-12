from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

@dataclass
class CategoryMappings:
    HOME_OWNERSHIP: Dict[str, int] = None
    LOAN_INTENT: Dict[str, int] = None
    LOAN_GRADE: Dict[str, int] = None
    DEFAULT_ON_FILE: Dict[str, int] = None
    
    def __post_init__(self):
        self.HOME_OWNERSHIP = {
            "RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3
        }
        self.LOAN_INTENT = {
            "EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2,
            "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5
        }
        self.LOAN_GRADE = {
            "G": 0, "F": 1, "E": 2, "D": 3, "C": 4, "B": 5, "A": 6
        }
        self.DEFAULT_ON_FILE = {"N": 0, "Y": 1}

@dataclass
class LoanFeatures:
    target: Tensor
    categorical: Dict[str, Tensor]
    numerical: Dict[str, Tensor]

class LoanDataProcessor:
    
    def __init__(self):
        self.mappings = CategoryMappings()
        
    def process_categorical(self, row: pd.Series) -> Dict[str, Tensor]:
        return {
            'person_home_ownership': torch.tensor(
                self.mappings.HOME_OWNERSHIP[row['person_home_ownership']], 
                dtype=torch.long
            ),
            'loan_intent': torch.tensor(
                self.mappings.LOAN_INTENT[row['loan_intent']], 
                dtype=torch.long
            ),
            'loan_grade': torch.tensor(
                self.mappings.LOAN_GRADE[row['loan_grade']], 
                dtype=torch.long
            ),
            'cb_person_default_on_file': torch.tensor(
                self.mappings.DEFAULT_ON_FILE[row['cb_person_default_on_file']], 
                dtype=torch.long
            )
        }
    
    def process_numerical(self, row: pd.Series) -> Dict[str, Tensor]:
        numerical_cols = [
            'person_age', 'person_income', 'person_emp_length',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length'
        ]
        return {
            col: torch.tensor(row[col], dtype=torch.float32)
            for col in numerical_cols
        }

class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.processor = LoanDataProcessor()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> LoanFeatures:
        row = self.data.iloc[idx]
        return LoanFeatures(
            target=torch.tensor(row['loan_status'], dtype=torch.float32),
            categorical=self.processor.process_categorical(row),
            numerical=self.processor.process_numerical(row)
        )

class LoanCollator:
    @staticmethod
    def _stack_dict(dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Stack tensors in dictionary"""
        return {
            key: torch.stack([d[key] for d in dicts])
            for key in dicts[0].keys()
        }
    
    def __call__(self, items: List[LoanFeatures]) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        return {
            'target': torch.stack([x.target for x in items]),
            'cat_features': self._stack_dict([x.categorical for x in items]),
            'numeric_features': self._stack_dict([x.numerical for x in items])
        }

def load_loan_data() -> Tuple[LoanDataset, LoanDataset]:
    data_dir = Path("data")
    train_data = pd.read_csv(data_dir / "loan_train.csv")
    test_data = pd.read_csv(data_dir / "loan_test.csv")
    
    return LoanDataset(train_data), LoanDataset(test_data)