import torch
import torch.nn as nn
from rmsnorm import RMSNorm

def test_rmsnorm():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 3
    hidden_dim = 4

    x = torch.randn(batch_size, seq_len, hidden_dim)

    our_rmsnorm = RMSNorm(hidden_dim)

    torch_rmsnorm = nn.RMSNorm(hidden_dim)

    torch_rmsnorm.weight.data.copy_(our_rmsnorm.weight.data)
    
    our_output = our_rmsnorm(x)
    torch_output = torch_rmsnorm(x)
    
    is_close = torch.allclose(our_output, torch_output, rtol=1e-5, atol=1e-5)
    max_diff = (our_output - torch_output).abs().max().item()
    
    print(f"Outputs are close: {is_close}")
    print(f"Maximum difference: {max_diff}")
    
    x_2d = torch.randn(batch_size, hidden_dim)
    our_output_2d = our_rmsnorm(x_2d)
    torch_output_2d = torch_rmsnorm(x_2d)
    
    is_close_2d = torch.allclose(our_output_2d, torch_output_2d, rtol=1e-5, atol=1e-5)
    max_diff_2d = (our_output_2d - torch_output_2d).abs().max().item()
    
    print(f"\n2D input - Outputs are close: {is_close_2d}")
    print(f"2D input - Maximum difference: {max_diff_2d}")

if __name__ == "__main__":
    test_rmsnorm()
