import torch
from torch.optim import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                
                beta1, beta2 = group['betas']
                lr = group['lr']
                weight_decay = group['weight_decay']
                
                state['step'] += 1
                
                update = (1 - beta1) * grad + beta1 * state['momentum']
                
                update = torch.sign(update)
                
                state['momentum'] = (1 - beta2) * grad + beta2 * state['momentum']
                
                if weight_decay != 0:
                    update = update + weight_decay * p.data
                
                update = update * lr
                
                p.data.add_(-update)
                
        return loss
