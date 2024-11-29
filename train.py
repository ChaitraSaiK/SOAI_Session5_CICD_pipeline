import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def save_augmentation_examples(train_transform, num_examples=5):
    """Save examples of original and augmented images"""
    # Create directory for visualization
    os.makedirs('visualization', exist_ok=True)
    
    # Basic transform for original images
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load some training examples
    dataset = datasets.MNIST('data', train=True, download=True, transform=None)
    
    plt.figure(figsize=(10, 4))
    for idx in range(num_examples):
        # Get original image
        img, label = dataset[idx]
        
        # Create original and augmented versions
        orig_img = basic_transform(img)
        aug_img = train_transform(img)
        
        # Plot original
        plt.subplot(2, num_examples, idx + 1)
        plt.imshow(orig_img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Original {label}')
        
        # Plot augmented
        plt.subplot(2, num_examples, num_examples + idx + 1)
        plt.imshow(aug_img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Augmented {label}')
    
    plt.tight_layout()
    plt.savefig('visualization/augmentation_examples.png')
    plt.close()

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
    
    # Save examples of augmented images
    save_augmentation_examples(train_transform)
    
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