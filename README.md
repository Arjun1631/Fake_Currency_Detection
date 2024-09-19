# Fake Currency Detection

## Overview

This Fake Currency Detection project aims to classify whether a given banknote is genuine or counterfeit based on specific features extracted from the currency images. The project leverages machine learning algorithms to identify fake currencies with high accuracy. The dataset provided contains various statistical features computed from the images of banknotes.

## Dataset

The dataset consists of features that describe certain statistical properties of the currency images. It includes 5 attributes:

- **Variance**: Variance of the wavelet-transformed image (continuous)
- **Skewness**: Skewness of the wavelet-transformed image (continuous)
- **Curtosis**: Curtosis of the wavelet-transformed image (continuous)
- **Entropy**: Entropy of the image (continuous)
- **Class**: Binary classification (0 for authentic, 1 for fake)

The dataset contains a mix of genuine and counterfeit currency samples, allowing us to train a model that can predict if a currency note is fake.

## Project Workflow

1. **Data Preprocessing**:
    - Load the dataset and handle missing or erroneous data (if any).
    - Split the data into training and testing sets.
    - Normalize or standardize the features to improve model performance.

2. **Exploratory Data Analysis (EDA)**:
    - Visualize the distributions of features.
    - Analyze the correlation between different features and the target class.
    - Visualize relationships between features using scatter plots and pair plots.

3. **Modeling**:
    - Implement different classification models such as:
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Gradient Boosting Classifier
        - XGBoost
        
4. **Model Evaluation**:
    - Use performance metrics such as:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Confusion Matrix
    - Perform cross-validation and hyperparameter tuning to improve model accuracy.

5. **Prediction**:
    - Use the trained model to classify whether a given currency note is genuine or fake based on its features.

## Requirements

- **Python 3.x**
- **Libraries**:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `scikit-learn`
    - `xgboost`

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fake-currency-detection.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    python main.py
    ```

## Results

- **Model Performance**: The model will output metrics such as accuracy, precision, and recall.
- **Prediction Example**: Use the trained model to predict the class of new currency notes, determining whether they are genuine or counterfeit.

## Conclusion

This project successfully demonstrates the use of machine learning techniques for detecting counterfeit banknotes. By applying various classification algorithms, we can accurately distinguish between genuine and fake currency notes based on their features. This solution could be scaled and deployed to assist in real-world counterfeit detection systems.
