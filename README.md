# Kidney Disease Classification from CT Scans using Deep Learning (ResNet18)- Streamlit

This project focuses on binary classification of CT scans of kidneys to identify Healthy and Abnormal (cyst, stone, tumor).

The core idea is to use transfer learning in medical imaging, where data is often limited and in single channel, to build a quality medical tool.



# Problem Overview

CT imaging is a primary tool used in identifying organ diseases but manually reviewing thousands of slices per patient is impractical and inefficient.

The model classifies each scan into two categories:
- Healthy (normal)
- Abnormal (cyst, stone, tumor)

The biggest challenge was to adapt a model which is pretrained on natural RGB images (ImageNet) to grayscale data while handling class imbalance of clinical datasets.



# Dataset Structure

The pipeline uses a dataset with four categories:

- Normal: 13460 images
- Cyst: 3709 images
- Stone: 1377 images
- Tumor: 2283 images

For binary classification cyst, stone and tumor classes are merged into a single category called *Abnormal* in order to prioritize abnormality detection over identifying the disease.



# Data Pipeline and Preprocessing

These steps are taken to ensure that the model could generalize to noisy medical scans.

- **Organized partitioning**: A custom *organize_data* script is implemented to split the dataset into train,val and test while maintaining class distribution.

- **Grayscale conversion**: Since Ct scans are single-channel, images were loaded in grayscale and expanded to a (H, W, 1) format.

- **Medical data augmentation**: Used the Albumentations library to simulate variations:
                                 
                                 - *vertical/horizontal flips*
                                 - *random brightness/contrast*
                                 - *coarse dropout*




# Model Architecture

The model uses a modified ResNet18 backbone to match these requirements:
 
-*Grayscale adaptation*: The first convolutional layer was modified from 3 input channels  (RGB) to 1 (Grayscale). I took the mean of the ImageNet weights across the channel dimension.

-*Fine tuning*: I froze the early layers and only un-froze Layer 4 and the Fully Connected (FC) head to prevent forgetting of low level feature detection.

-*Binary head*: Replaced the final 1000-node layer with a single-node linear layer followed by a Sigmoid activation for binary probability output.



# Training & Optimization

Implemented using pyTorch Lightning for a clean and reproducible pipeline:

-**Imbalance handling**: Used BCEWithLogitsLoss which penalizes the model more for missing an "Abnormal" case.

-**Learning rate scheduling**: Used ReduceLROnPlateau , which halves the learning rate if the validation loss plateaus for 2 epochs.

-**Callbacks**: Integrated ModelCheckpoint to save only the version with the highest Validation AUC and EarlyStopping to prevent overfitting.




# Performance Metrics

The model was evaluated primarily on ROC-AUC rather than simple accuracy, as AUC is a better on showing how well the model separates "Healthy" from "Abnormal".

**Metric Goal**: Maximize Recall for the Abnormal class to minimize False Negatives (missing a disease).




# Learning Outcomes

- Learned how to modify pre trained CNN architecture to accept single channel(grayscale) data without losing pre-trained weights.

- Gained experience in medical data engineering.

- Improved understanding of PyTorch Lightning for managing complex training loops, checkpointing.




# Limitations

**Slice-based vs. Volume-based:** This model analyzes 2D slices but in real clinical situations a 3D image is used for better diagnosis.

**Data source:** Used datset is relatively clean compared to real world CT scans are with noise that may degrade performance.

**Binary factor:** The model identifies abnormality but not distinguish the disease.



# NOTES:

- This project is for educational purposes only and is not medically accurate. It should not be used for clinical decisions.

- I developed this as a personal project to apply Deep Learning to medical computer vision challenges.




# Acknowledgements

- Data source: CT-KIDNEY-DATASET on Kaggle

- Backbone: ResNet18 (Torchvision Weights)  

