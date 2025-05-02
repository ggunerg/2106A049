GUI2

## Description:

This Python-based graphical user interface (GUI) application enables users to perform **dimensionality reduction**, **visualization**, and **classification** on uploaded datasets. The interface is designed for ease of use in educational or research settings and integrates key machine learning techniques such as PCA, LDA, t-SNE, and UMAP.

## Features:

* Upload CSV datasets with labeled data.
* Select and apply dimensionality reduction methods:

  * PCA (Principal Component Analysis)
  * LDA (Linear Discriminant Analysis)
  * t-SNE (t-distributed Stochastic Neighbor Embedding)
  * UMAP (Uniform Manifold Approximation and Projection)
* Choose classification algorithms:

  * SVM (Support Vector Machine)
  * KNN (K-Nearest Neighbors)
  * Decision Tree
  * Random Forest
  * Naive Bayes
* Visualize results in 2D and 3D.
* Confusion matrix, accuracy, precision, recall, and F1-score output.
* Easy-to-use Tkinter GUI.

## Dependencies:

* Python 3.x
* tkinter
* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* umap-learn

To install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn
```

## How to Run:

1. Ensure Python 3 is installed on your system.
2. Install the required packages listed above.
3. Run the script:

   ```bash
   python 2106A049_GUI2.py
   ```
4. Use the GUI to upload your dataset and begin analysis.

## Notes:

* The dataset must contain labeled data (i.e., a target column for classification).
* You may need to adjust column indices or names depending on your dataset format.

## Author:

GUI1

# Advanced Machine Learning GUI

## Overview
This is an advanced machine learning graphical user interface (GUI) developed for students and researchers to explore and experiment with various machine learning algorithms, preprocessing techniques, and model evaluation methods.

## Features

### Data Management
- Support for multiple built-in datasets:
  * Iris Dataset
  * Breast Cancer Dataset
  * Digits Dataset
  * Boston Housing Dataset
  * MNIST Dataset
- Custom dataset loading via CSV
- Data preprocessing options:
  * Train-test split configuration
  * Multiple scaling techniques:
    - No Scaling
    - Standard Scaling
    - Min-Max Scaling
    - Robust Scaling

### Machine Learning Algorithms

#### Regression
- Linear Regression
  * Multiple loss function options
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Huber Loss
- Support Vector Regression (SVR)
  * Kernel selection (linear, RBF, polynomial)
  * Hyperparameter tuning

#### Classification
- Naive Bayes
  * Configurable var_smoothing
  * Prior probability options
- Support Vector Machine (SVM)
  * Kernel selection
  * Loss function selection
    - Hinge Loss
    - Cross-Entropy
- Decision Tree
- Random Forest
- K-Nearest Neighbors

### Deep Learning
- Multi-Layer Perceptron (MLP)
  * Customizable layer configuration
  * Training parameter controls
- Convolutional Neural Network (CNN) framework
- Recurrent Neural Network (RNN) framework

### Visualization
- Interactive visualization of:
  * Model predictions
  * Training history
  * Performance metrics
- Support for:
  * Regression prediction plots
  * Classification decision boundaries
  * Dimensionality reduction visualization

## Prerequisites

### Python Dependencies
- PyQt6
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorFlow
- Keras

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ml-gui.git
cd ml-gui
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install required packages
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python ml_gui.py
```

## Usage Guide

1. Load a Dataset
   - Select from built-in datasets
   - Upload a custom CSV file
   - Configure train-test split
   - Apply preprocessing scaling

2. Choose Machine Learning Algorithm
   - Navigate through tabs
   - Configure algorithm-specific parameters
   - Train and evaluate models

3. Analyze Results
   - View performance metrics
   - Explore visualization plots
   - Compare different model configurations

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

