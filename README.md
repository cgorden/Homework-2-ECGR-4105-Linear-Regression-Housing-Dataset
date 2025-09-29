# Housing Price Prediction using Linear Regression with Gradient Descent

A comprehensive implementation of linear regression with gradient descent from scratch for predicting housing prices. This project explores the effects of feature selection, data preprocessing, and regularization on model performance.

## Overview

This project implements a complete machine learning pipeline to predict housing prices using a custom gradient descent algorithm. The implementation compares different approaches to feature engineering and preprocessing to understand their impact on model performance.

### Key Features

- **From-scratch gradient descent implementation** - No built-in ML libraries for core algorithm
- **Multiple feature sets** - Comparison of 5-feature vs 11-feature models
- **Data preprocessing comparison** - Normalization vs Standardization analysis
- **Ridge regularization** - Parameter penalty implementation to prevent overfitting
- **Comprehensive visualization** - Training/validation loss plots for all experiments
- **Performance evaluation** - R² scoring and parameter analysis

## Dataset

The project uses a US Housing dataset with the following features:

**Numerical Features:**
- `area` - House area in square feet
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms  
- `stories` - Number of stories
- `parking` - Number of parking spaces

**Categorical Features (converted to binary):**
- `mainroad` - Whether house is on main road (yes=1, no=0)
- `guestroom` - Whether house has guest room (yes=1, no=0)
- `basement` - Whether house has basement (yes=1, no=0)
- `hotwaterheating` - Whether house has hot water heating (yes=1, no=0)
- `airconditioning` - Whether house has air conditioning (yes=1, no=0)
- `prefarea` - Whether house is in preferred area (yes=1, no=0)

**Target Variable:**
- `price` - Housing price (to be predicted)

## Project Structure

```
housing-prediction/
│
├── Housing.csv                 # Dataset file
├── housing_prediction.py       # Main implementation
└── README.md                  # This file
```

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
seaborn
```

## Installation & Setup

### For Google Colab (Recommended)

1. Upload your `Housing.csv` file to Google Drive
2. Mount Google Drive in your Colab notebook
3. Update the file path in the code:
   ```python
   FilePath = '/content/drive/MyDrive/path/to/your/Housing.csv'
   ```
4. Install required packages (usually pre-installed in Colab):
   ```python
   !pip install numpy pandas matplotlib scikit-learn seaborn
   ```

### For Local Environment

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/housing-prediction.git
   cd housing-prediction
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```

3. Place your `Housing.csv` file in the project directory

4. Update the file path in the code:
   ```python
   FilePath = './Housing.csv'
   ```

## Usage

Run the complete analysis:

```python
python housing_prediction.py
```

The script will execute all experiments sequentially and generate:
- Console output with training progress and results
- Visualization plots for each problem
- Summary tables comparing all approaches

## Experiment Structure

### Problem 1: Baseline Models
- **1a**: 5-feature model (area, bedrooms, bathrooms, stories, parking)
- **1b**: 11-feature model (all features)
- **Learning rates tested**: 0.01, 0.03, 0.05, 0.1
- **Goal**: Establish baseline performance and identify optimal learning rates

### Problem 2: Data Preprocessing
- **2a**: 5-feature model with normalization vs standardization
- **2b**: 11-feature model with normalization vs standardization
- **Preprocessing methods**:
  - Normalization: Min-Max scaling to [0,1]
  - Standardization: Z-score normalization (mean=0, std=1)
- **Goal**: Compare preprocessing impact on convergence and performance

### Problem 3: Regularization
- **3a**: 5-feature model with Ridge regularization
- **3b**: 11-feature model with Ridge regularization
- **Lambda values tested**: 0.001, 0.01, 0.1, 1.0
- **Goal**: Evaluate regularization effects on overfitting

## Key Implementation Details

### LinearRegression Class
- **Zero initialization**: All weights and bias start at zero (as required)
- **Custom gradient descent**: Implements weight updates using computed gradients
- **Ridge regularization**: Adds L2 penalty term to loss function (training only)
- **Error handling**: Includes convergence monitoring and numerical stability checks

### Gradient Computation
```python
GradW = -(2/NumSamples) * np.dot(XData.T, (YTrue - YPred))
GradB = -(2/NumSamples) * np.sum(YTrue - YPred)
```

### Loss Function
- **Mean Squared Error**: Primary loss metric
- **Ridge penalty**: λ * Σ(weights²) added during training only
- **Validation loss**: MSE without regularization term

## Results Interpretation

### Performance Metrics
- **R² Score**: Coefficient of determination (higher = better, max = 1.0)
- **MSE Loss**: Mean squared error (lower = better)
- **Training vs Validation**: Used to detect overfitting

### Expected Outcomes
1. **Feature Impact**: More features generally improve performance
2. **Preprocessing Benefits**: Scaled data converges faster and more stably
3. **Regularization Trade-off**: Reduces overfitting but may decrease training performance
