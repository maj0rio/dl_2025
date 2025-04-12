import pandas as pd
import numpy as np
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from arch import RegularizedModel
from loan_dataset import LoanCollator, load_loan_data
from plots import VisualizationManager

@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    hidden_size: int
    drop_p: float
    lr: float
    weight_decay: float
    num_epochs: int
    
    @classmethod
    def from_yaml(cls, config_path: str) -> Optional['TrainingConfig']:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                params = config.get('hyperparameters', {})
                if not params:
                    print('Failed to load config')
                    return None
                print(f'Training model with config:\n{params}')
                return cls(**params)
        except Exception as e:
            print(f'Error loading config: {e}')
            return None

class TrainingManager:
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.viz = VisualizationManager()
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = RegularizedModel(
            hidden_size=config.hidden_size,
            drop_p=config.drop_p
        ).to(self.device)
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_roc = 0
        
        with tqdm(total=len(train_loader), desc='Training') as pbar:
            for batch in train_loader:
                self.optimizer.zero_grad()
                

                cat_features = {k: v.to(self.device) for k, v in batch['cat_features'].items()}
                numeric_features = {k: v.to(self.device) for k, v in batch['numeric_features'].items()}
                target = batch['target'].to(self.device)
                
                outputs = self.model(
                    cat_features=cat_features,
                    numeric_features=numeric_features
                )
                loss = self.criterion(outputs, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                target_np = target.cpu().numpy()
                fpr, tpr, _ = roc_curve(target_np, probs)
                total_roc += auc(fpr, tpr)
                
                pbar.update(1)
                
        return total_loss, total_roc / len(train_loader)
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_roc = 0
        
        with tqdm(total=len(val_loader), desc='Validation') as pbar:
            for batch in val_loader:
                cat_features = {k: v.to(self.device) for k, v in batch['cat_features'].items()}
                numeric_features = {k: v.to(self.device) for k, v in batch['numeric_features'].items()}
                target = batch['target'].to(self.device)
                
                outputs = self.model(
                    cat_features=cat_features,
                    numeric_features=numeric_features
                )
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                target_np = target.cpu().numpy()
                fpr, tpr, _ = roc_curve(target_np, probs)
                total_roc += auc(fpr, tpr)
                
                pbar.update(1)
                
        return total_loss, total_roc / len(val_loader)
    
    def train(self) -> None:
        train_dataset, val_dataset = load_loan_data()
        collator = LoanCollator()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=collator
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collator
        )
        
        train_losses = []
        val_losses = []
        train_rocs = []
        val_rocs = []
        
        for epoch in range(self.config.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.config.num_epochs}')
            
            train_loss, train_roc = self._train_epoch(train_loader)
            train_losses.append(train_loss)
            train_rocs.append(train_roc)
            print(f'Training - Loss: {train_loss:.4f}, ROC-AUC: {train_roc:.4f}')
            
            val_loss, val_roc = self._evaluate(val_loader)
            val_losses.append(val_loss)
            val_rocs.append(val_roc)
            print(f'Validation - Loss: {val_loss:.4f}, ROC-AUC: {val_roc:.4f}')
        
        config_dict = self.config.__dict__
        self.viz.plot_training_history(train_losses, val_losses, config_dict)
        self.viz.plot_metric_history(val_rocs, config_dict)
        
        all_probs = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                cat_features = {k: v.to(self.device) for k, v in batch['cat_features'].items()}
                numeric_features = {k: v.to(self.device) for k, v in batch['numeric_features'].items()}
                target = batch['target'].to(self.device)
                
                outputs = self.model(
                    cat_features=cat_features,
                    numeric_features=numeric_features
                )
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        final_roc_auc = auc(fpr, tpr)
        self.viz.plot_roc_curve(fpr, tpr, final_roc_auc)

def main():
    config = TrainingConfig.from_yaml('config.yaml')
    if config is None:
        return
    
    trainer = TrainingManager(config)
    trainer.train()

if __name__ == '__main__':
    main()