[![ML Pipeline](https://github.com/ChaitraSaiK/SOAI_Session5_CICD_pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/ChaitraSaiK/SOAI_Session5_CICD_pipeline/actions/workflows/ml-pipeline.yml)

# Description:

This project deals with a machine learning CI/CD pipeline for MNIST classification using PyTorch.

A CI/CD pipeline is a series of automated steps that streamline the process of creating, testing, and deploying software:

CI - Stands for continuous integration, where developers frequently merge code changes into a central repository. This allows for early detection of issues.

CD - Stands for continuous delivery or continuous deployment, which automates the release of the application to its intended environment.

Benefits of CI/CD pipelines:
- Faster feedback on code changes
- Early detection of issues
- Reduced manual errors
- Faster software delivery
- Improved collaboration among team members
- Consistent and reliable deployment process

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


