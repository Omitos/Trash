# Automated Waste Sorting System

This repository contains a deep learning project focused on classifying waste images into categories such as **Recyclable**, **Compost**, and **Landfill** using convolutional neural networks (CNNs). The goal is to build an image classification system that can automatically identify waste types to assist in efficient and environmentally-friendly sorting.

---

## Project Overview

This project leverages computer vision and transfer learning to automate the classification of waste materials. It uses models such as **custom CNNs** and **pretrained architectures (e.g., ResNet50)**, trained on the **TrashNet dataset** (or similar).

### Key Features:
- **Waste Categories**: Recyclable, Compost, Landfill (including subtypes like metal, plastic, paper, glass, etc.)
- **Image Preprocessing**: Resizing, normalization, and augmentation
- **Modeling Techniques**: CNN from scratch, Transfer Learning (ResNet50)
- **Evaluation Tools**: Confusion matrix, classification report, and visual feedback

---

## Skills Gained

- CNN design and tuning
- Data preprocessing and augmentation
- Transfer learning implementation
- Model evaluation with classification metrics
- Deployment with Python and PyTorch

---

## Requirements

To run the project, install Python 3.7+ and the following packages:

```bash
pip install -r requirements.txt
```

## Key Libraries Used

- **torch**, **torchvision** – For building and training deep learning models, and handling datasets and image transforms.
- **matplotlib**, **seaborn** – For visualizing training metrics, confusion matrices, and sample predictions.
- **PIL (Python Imaging Library)** – For loading and preprocessing images.
- **scikit-learn (sklearn)** – For evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.

## Project Structure
```bash
.
├── data_trash/ # Contains images of different waste types
│ └── [glass, plastic, paper, etc.]
├── data-resezied/ # Contains images of different waste types
│ └── [cardboard, glass, plastic, paper, etc.]
├── Version 1/ # 1st version of model using data from TrashNet
│ └── Test.ipynb 
│ └── TrashApp.py
│ └── trashnet.pt
├── Version 2/ # 2nd version of model using data from gathering by own hands
│ └── Testv2.ipynb
│ └── TrashAppv2.py
│ └── trashnetV2.pt
├── doc.pdf # Fully written documentation about project
├── Poster.png # Project poster for presentation
├── requirements.txt # Required packages
└── README.md # Project documentation
```
## Dataset

Firstly, the project uses the **TrashNet** dataset, which contains labeled images of common waste items across several categories:

- Glass  
- Plastic  
- Cardboard  
- Metal  
- Paper  
- Trash (landfill)  

Secondly, the project uses the **data gathering from own hands** dataset, which contains labeled images of common waste items across several categories:
- Glass  
- Plastic  
- Cans 
- Paper  
- Food

Each image is preprocessed and labeled accordingly to train the model effectively.

---

## Model Pipeline

### Preprocessing
- Resize images to **224x224**
- Normalize using **ImageNet mean and standard deviation**
- Apply data augmentation:
  - Random horizontal flips
  - Random rotations

### Architecture Options
- **Option 1**: Custom CNN  
  - Layers: `Conv2D → ReLU → MaxPool → FC`
- **Option 2**: Pre-trained **ResNet50**  
  - Fine-tuned last layers for classification

### ️ Training
- **Loss Function**: `NLLLoss`
- **Optimizer**: `Adam`
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-score

## Evaluation Results

### Custom CNN
- **Accuracy**: ~77%
- Performs well on classes like **glass** and **paper**
- Some confusion observed between **plastic** and **metal** due to visual similarity

### ResNet50 (Transfer Learning)
- **Accuracy**: ~85%
- Shows **better generalization** across all classes
- **Faster convergence** and reduced overfitting

### Sample Classification Report for Version 1


                  precision    recall  f1-score   support

    cardboard       0.85      0.92      0.88        36
    glass           0.91      0.72      0.81        40
    metal           0.75      0.91      0.82        43
    paper           0.87      0.91      0.89        58
    plastic         0.88      0.85      0.86        60
    trash           0.70      0.47      0.56        15

    accuracy                            0.84        252
    macro avg       0.83      0.80      0.80        252
    weighted avg    0.84      0.84      0.84        252


### Sample Classification Report for Version 2


                   precision    recall  f1-score   support

    cans             0.43        0.75     0.55        4
    food             0.82        0.82     0.82       11
    glass            0.90        0.90     0.90       10
    paper            0.92        0.92     0.92       12
    plastic          0.90        0.69     0.78       13

    accuracy                              0.82       50
    macro avg        0.79        0.82     0.79       50
    weighted avg     0.85        0.82     0.83       50


## Known Issues

- **Model paths**: Must be **absolute** if running inside virtual environments to avoid `FileNotFoundError`.
-️ **Deprecation warnings**: 
  - Messages related to `pretrained=True` (now replaced with `weights=...`) in `torchvision.models` are safe to ignore.
  - You can update the code to follow the latest PyTorch API if desired.

---

## Authors

- **Omar Aitimbet** – *ID: 220103131*
- **Dinmukhamed Sapybek** – *ID: 220103053*

---

## References

- [TrashNet Dataset](https://github.com/garythung/trashnet)
- [PyTorch Transfer Learning Documentation](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Scikit-learn Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
