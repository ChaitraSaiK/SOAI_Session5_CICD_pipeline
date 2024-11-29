# ML CI/CD Pipeline

A machine learning CI/CD pipeline for MNIST classification using PyTorch.

## Project Structure

.
├── .github
│ └── workflows
│ └── ml-pipeline.yml
├── model
│ ├── init.py
│ └── network.py
├── tests
│ └── test_model.py
├── models/ # Created during training
│ └── .gitkeep # Empty file to maintain folder structure
├── .gitignore
├── README.md
├── requirements.txt
└── train.py

## Setup and Running

1. Create a virtual environment: pip install -r requirements.txt 
2. Activate virtual environment: venv\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Run training: python train.py
5. Run tests: pytest tests/

## Model Architecture
- Simple CNN with 2 convolutional layers
- Input: 28x28 MNIST images
- Output: 10 classes (digits 0-9)
- Training: 1 epoch
- Validation accuracy: >80%

## CI/CD Pipeline
- Automated training on push
- Model validation tests
- Artifact storage of trained models


