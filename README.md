# Swin-Brain-Age-Estimation

# Vision Transformer for MRI-based Brain Age Estimation

This project implements a SWIN Transformer based approach for estimating brain age from MRI scans, achieving superior performance compared to traditional CNN architectures.

## Overview

The system performs ternary classification of brain MRI scans into three age groups:
- Children ages 3-5
- Children ages 7-12  
- Adults

Using a Swin Transformer architecture with transfer learning and fine-tuning, we achieved 92.5% classification accuracy on the test dataset, outperforming traditional CNN approaches like ResNet18 (85%) and VGG16 (82.5%).

## Dataset

The dataset used is from OpenNeuro and consists of:
- 65 subjects aged 3-5
- 57 subjects aged 7-12 
- 33 adult subjects

MRI scans were taken while subjects watched "Partly Cloudy," a Disney Pixar short film. To address the small dataset size, data augmentation was performed by flipping all original images 180 degrees, effectively doubling the dataset size.

## Model Architecture

We implemented and compared three architectures:
1. VGG16
2. ResNet18  
3. Swin Vision Transformer

All models were pretrained on ImageNet and fine-tuned on our dataset. The Swin ViT architecture particularly excelled by combining the convolution sliding window approach with an attention mechanism to create hierarchical feature maps.

## Results

Model performance comparison:

| Model     | Classification Accuracy |
|-----------|------------------------|
| VGG-16    | 82.5%                 |
| ResNet-18 | 85.0%                 |
| Swin ViT  | 92.5%                 |

Detailed metrics for the Swin ViT model:

| Label  | Precision | Recall | F1-score |
|--------|-----------|---------|-----------|
| infant | 0.8462    | 0.9167  | 0.8800    |
| child  | 0.9412    | 0.8889  | 0.9143    |
| adult  | 1.0000    | 1.0000  | 1.0000    |

## Training Details

- Hardware: Google Tesla T4 GPU with 16GB RAM
- Epochs: 25  
- Batch size: 4
- Loss function: Cross entropy
- Optimizer configurations:
  - ResNet18: SGD with learning rate 0.001
  - VGG16: Adam with learning rate 0.0001
  - Swin ViT: AdamW with learning rate 0.00001 and learning rate decay of 0.05 every 5 epochs

## Key Findings

1. The Swin ViT architecture achieved superior performance despite the relatively small dataset size
2. Transfer learning and fine-tuning proved effective for the medical imaging domain
3. Data augmentation helped address the limited dataset size
4. The attention mechanism showed particular strength in capturing relevant features for age estimation

## Future Work

Potential areas for improvement and exploration:
- Experiment with larger datasets
- Compare computational efficiency between architectures
- Explore additional data augmentation techniques
- Investigate model interpretability
- Test on different medical imaging modalities

## Citations

The project builds upon several key papers:
1. Dosovitskiy et al. (2020) - Original ViT architecture
2. Liu et al. (2021) - Swin Transformer
3. Asiri et al. (2023) - Fine-tuning transformers for medical imaging
4. Additional references available in the full paper

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib (for visualization)
