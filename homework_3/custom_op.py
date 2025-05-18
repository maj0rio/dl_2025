import torch
from torch.autograd import Function

class ExpPlusCos(Function):
    @staticmethod
    def forward(ctx, x, y):
        exp_x = torch.exp(x)
        cos_y = torch.cos(y)
        result = exp_x + cos_y

        ctx.save_for_backward(x, y, exp_x)

        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, exp_x = ctx.saved_tensors
        
        grad_x = grad_output * exp_x  # d/dx(e^x) = e^x
        grad_y = grad_output * (-torch.sin(y))  # d/dy(cos(y)) = -sin(y)
        
        return grad_x, grad_y
