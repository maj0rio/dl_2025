import torch
import torch.nn as nn
from lion import Lion

def test_lion_optimizer():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    optimizer = Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100
    
    print("Training with Lion optimizer:")
    for epoch in range(n_epochs):
        outputs = model(x)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    print("\nTraining with Adam optimizer for comparison:")
    model_adam = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        outputs = model_adam(x)
        loss = criterion(outputs, y)
        
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    test_lion_optimizer()
