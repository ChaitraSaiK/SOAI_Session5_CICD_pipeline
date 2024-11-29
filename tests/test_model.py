import torch
import pytest
from torchvision import transforms, datasets
from model.network import SimpleCNN
import os
import glob

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    model = SimpleCNN()
    
    # Test 1: Check if model accepts 28x28 input
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), "Output shape is incorrect"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")
    
    # Test 2: Check number of parameters
    num_params = count_parameters(model)
    assert num_params < 100000, f"Model has {num_params} parameters, should be less than 100000"
    
    # Test 3: Check output dimension
    assert output.shape[1] == 10, "Model should have 10 output classes"

def test_model_accuracy():
    device = torch.device("cpu")
    
    # Load the latest trained model
    model_files = glob.glob('models/model_*.pth')
    if not model_files:
        pytest.skip("No trained model found")
    
    latest_model = max(model_files, key=os.path.getctime)
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    
    # Load test dataset with only normalization (no augmentation for testing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 