# Animal_Detection_using_Machine_learning
Animal detection and classification of African wildlife (Buffalo, Elephant, Rhino, Zebra) using deep learning with PyTorch Lightning and Hugging Face’s YOLOS-tiny model.


**Automated Animal Detection Using Machine Learning**

This project implements an **object detection pipeline** to automatically detect and classify African wildlife (Buffalo, Elephant, Rhino, and Zebra) using **deep learning**. It leverages **transformers, PyTorch Lightning, and Hugging Face’s YOLOS-tiny model** for efficient detection and training on custom datasets.

**Project Overview**

The goal of this project is to **automatically detect and classify wild animals from images** using **state-of-the-art object detection models**.

The workflow includes:

1. **Dataset Preparation** – Images and YOLO-style bounding box annotations of animals (Zebra, Buffalo, Rhino, Elephant).
2. **Data Processing** – Custom PyTorch Dataset class for reading, resizing, and normalizing images.
3. **Model Training** – Fine-tuning Hugging Face’s **YOLOS-tiny** object detection transformer using **PyTorch Lightning**.
4. **Evaluation** – Using **Mean Average Precision (mAP)** metrics at multiple IoU thresholds.
5. **Visualization** – Bounding box predictions drawn on sample images for visual inspection.

**Tools & Technologies Used**

- **Python 3.x** → Core programming language.
- **PyTorch** → Deep learning framework for building datasets, models, and training.
- **Torchvision** → Image transformations and utilities for working with vision datasets.
- **PyTorch Lightning** → High-level framework for structuring and training deep learning models.
- **Transformers (Hugging Face)** → Pretrained models like **YOLOS-tiny** for object detection.
- **torchmetrics (detection)** → For computing **Mean Average Precision (mAP)** and evaluation metrics.
- **timm** → Image models collection used as backbones for transformers.
- **Datasets (Hugging Face)** → For data handling and preprocessing pipelines.
- **Matplotlib & PIL** → Visualization of images, annotations, and model predictions.
- **Google Colab + Drive** → For training, dataset storage, and experiments.

**How to Run**

**1\. Install Dependencies**

!pip install huggingface_hub --upgrade

!pip install torchmetrics\[detection\]

!pip install transformers==4.26.0

!pip install -U pytorch-lightning

!pip install timm

!pip install datasets

**2\. Mount Google Drive (if using Colab)**

from google.colab import drive

drive.mount('/content/drive')

**3\. Train the Model**

from pytorch_lightning import Trainer

trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)

trainer.fit(mymodel)

**4\. Evaluate Model**

trainer.validate(mymodel)

trainer.test(mymodel)

**5\. Visualize Predictions**

import matplotlib.pyplot as plt

img = show_model_outputs()

plt.imshow(img)

**Results**

- The model predicts bounding boxes and labels for **Buffalo, Elephant, Rhino, Zebra**.
- Evaluation uses **Mean Average Precision (mAP)** at IoU thresholds (0.5, 0.75, etc.).
- Visualization confirms accurate localization and classification on test data.
