import torch
from custom_op import ExpPlusCos

def test_custom_op():
    torch.manual_seed(42)

    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)
    
    x_std = x.clone().detach().requires_grad_(True)
    y_std = y.clone().detach().requires_grad_(True)
    
    custom_result = ExpPlusCos.apply(x, y)
    
    std_result = torch.exp(x_std) + torch.cos(y_std)
    
    is_close_forward = torch.allclose(custom_result, std_result, rtol=1e-5, atol=1e-5)
    max_diff_forward = (custom_result - std_result).abs().max().item()
    
    print("Forward pass comparison:")
    print(f"Results are close: {is_close_forward}")
    print(f"Maximum difference: {max_diff_forward}")
    
    custom_result.sum().backward()
    std_result.sum().backward()
    
    is_close_grad_x = torch.allclose(x.grad, x_std.grad, rtol=1e-5, atol=1e-5)
    is_close_grad_y = torch.allclose(y.grad, y_std.grad, rtol=1e-5, atol=1e-5)
    
    max_diff_grad_x = (x.grad - x_std.grad).abs().max().item()
    max_diff_grad_y = (y.grad - y_std.grad).abs().max().item()
    
    print("\nGradient comparison:")
    print(f"Gradients for x are close: {is_close_grad_x}")
    print(f"Maximum difference in x gradients: {max_diff_grad_x}")
    print(f"Gradients for y are close: {is_close_grad_y}")
    print(f"Maximum difference in y gradients: {max_diff_grad_y}")
    
    print("\nTesting with different shapes:")
    shapes = [(2,), (2, 3), (2, 3, 4)]
    
    for shape in shapes:
        x = torch.randn(shape, requires_grad=True)
        y = torch.randn(shape, requires_grad=True)
        
        x_std = x.clone().detach().requires_grad_(True)
        y_std = y.clone().detach().requires_grad_(True)
        
        custom_result = ExpPlusCos.apply(x, y)
        std_result = torch.exp(x_std) + torch.cos(y_std)
        
        custom_result.sum().backward()
        std_result.sum().backward()
        
        is_close = torch.allclose(custom_result, std_result, rtol=1e-5, atol=1e-5)
        is_close_grad = torch.allclose(x.grad, x_std.grad, rtol=1e-5, atol=1e-5) and \
                       torch.allclose(y.grad, y_std.grad, rtol=1e-5, atol=1e-5)
        
        print(f"\nShape {shape}:")
        print(f"Forward pass is correct: {is_close}")
        print(f"Backward pass is correct: {is_close_grad}")

if __name__ == "__main__":
    test_custom_op()
