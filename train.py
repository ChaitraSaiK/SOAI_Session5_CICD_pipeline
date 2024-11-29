import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from tqdm import tqdm

def train():
    # Set device (force CPU if CUDA is not available)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset with augmentations
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),  # Random rotation up to 15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random translation up to 10%
            scale=(0.9, 1.1),  # Random scaling between 90% and 110%
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2)  # Randomly erase parts of image
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss = 0.9 * running_loss + 0.1 * loss.item()  # Smoothed loss
        pbar.set_postfix({'loss': f'{running_loss:.4f}'})
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/model_{timestamp}.pth')
    
if __name__ == "__main__":
    train() 