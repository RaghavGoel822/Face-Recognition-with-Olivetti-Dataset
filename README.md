# Face Recognition with Olivetti Dataset

## Overview
This project implements a Face Recognition system using the Olivetti Faces dataset. It applies a global feature-based approach, primarily utilizing Principal Component Analysis (PCA) for feature extraction (Eigenfaces), and compares various machine learning algorithms to determine the most effective classification model.

## Dataset
The **Olivetti Faces Dataset** features:
- **400** face images taken between April 1992 and April 1994.
- **40** distinct individuals, with 10 images provided for each person.
- Variations in lighting, facial expressions (open/closed eyes, smiling/not smiling), and facial details (glasses/no glasses).
- Grayscale images with a resolution of 64x64 pixels.
- Pixel values are scaled to the [0, 1] interval.

## Methodology
The system pipeline consists of the following structure:
1. **Data Loading & Exploratory Data Analysis (EDA)**: Instantiating the dataset and visualizing the unique individuals to understand the underlying data distribution.
2. **Dimensionality Reduction (PCA)**: Applying Principal Component Analysis to generate *Eigenfaces*, drastically reducing the dimensionality of the images while preserving the most relevant variance.
3. **Model Training & Comparison**: Training multiple classification algorithms on the PCA-transformed data to establish a performance baseline:
   - Logistic Regression (LR)
   - Gaussian Naive Bayes (NB)
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier (DT)
   - Support Vector Machine (SVM)
4. **Validation & Optimization**: 
   - Assessing model robustness via K-Fold Cross-Validation.
   - Performing Leave-One-Out Cross-Validation (LOO CV), which is highly recommended for datasets with small sample sizes per class.
   - Fine-tuning hyperparameters for the top-performing model using `GridSearchCV`.
5. **Detailed Evaluation**: Visualizing findings using Confusion Matrices (via Seaborn heatmaps), extensive Classification Reports (Precision, Recall, F1-Score), and Multi-class Precision-Recall Curves leveraging `OneVsRestClassifier`.

## Key Results
- **Logistic Regression** achieved the best baseline performance with an accuracy of ~**93%**.
- Using **Leave-One-Out Cross-Validation**, the Logistic Regression model demonstrated an excellent mean accuracy of **96%**.
- Multi-class Precision-Recall validation resulted in an impressive **Average Precision** score of **0.97** (micro-averaged across all 40 classes).

## Requirements
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Usage
Simply open and run `FaceRecognition.ipynb` in a Jupyter environment. The notebook is self-contained and walks through data ingestion, preprocessing, visualization, model training, and detailed evaluation sequentially.
