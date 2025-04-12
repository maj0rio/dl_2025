from typing import Dict, List, Any
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

class VisualizationManager:
    def __init__(self, save_dir: str = 'experiments'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self.colors = {
            'train': '#2E86C1',  # Blue
            'val': '#E74C3C',    # Red
            'baseline': '#2C3E50' # Dark gray
        }
        
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.titlepad': 15,
            'font.size': 10,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
    def _setup_figure(self) -> None:
        plt.figure()
        
    def save_plot(self, filename: str) -> None:
        self.save_dir.mkdir(exist_ok=True)
        
        plt.savefig(
            self.save_dir / filename,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
    def plot_training_history(
        self,
        train_loss: List[float],
        val_loss: List[float],
        config: Dict[str, Any]
    ) -> None:
        self._setup_figure()
        
        epochs = range(1, len(train_loss) + 1)
        
        plt.plot(
            epochs, train_loss,
            color=self.colors['train'],
            label='Training Loss',
            linestyle='--',
            marker='o',
            markersize=4,
            alpha=0.8
        )
        
        plt.plot(
            epochs, val_loss,
            color=self.colors['val'],
            label='Validation Loss',
            linestyle='-',
            marker='x',
            markersize=4
        )
        
        plt.title('Training and Validation Loss')
        plt.xlabel(f'Epoch\nConfig: {config}')
        plt.ylabel('Loss')
        plt.legend()
        
        self.save_plot('loss_history.png')
        
    def plot_metric_history(
        self,
        metric_values: List[float],
        config: Dict[str, Any],
        metric_name: str = 'ROC-AUC'
    ) -> None:
        self._setup_figure()
        
        epochs = range(1, len(metric_values) + 1)
        plt.plot(
            epochs, metric_values,
            color=self.colors['val'],
            label=f'Validation {metric_name}',
            linewidth=2,
            marker='o',
            markersize=4
        )
        
        plt.title(f'{metric_name} Score Over Training')
        plt.xlabel(f'Epoch\nConfig: {config}')
        plt.ylabel(metric_name)
        plt.legend()
        
        self.save_plot('metric_history.png')
        
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float
    ) -> None:
        self._setup_figure()
        
        plt.plot(
            fpr, tpr,
            color=self.colors['val'],
            linewidth=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        
        plt.plot(
            [0, 1], [0, 1],
            color=self.colors['baseline'],
            linestyle='--',
            linewidth=1.5,
            label='Random Baseline'
        )
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        
        self.save_plot('roc_curve.png')
