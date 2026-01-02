# ✈️ Airline Passenger Satisfaction Prediction
A Machine Learning & Deep Learning Project


## Problem Statement
Customer satisfaction is a critical factor for airline companies. Understanding which service and operational factors influence passenger satisfaction helps airlines improve customer experience and business performance.

This project predicts whether a passenger is Satisfied or Not Satisfied based on demographic, travel, and service-related features using Machine Learning and Deep Learning models.


## Dataset Description
Source: Airline Passenger Satisfaction Dataset
Total Rows: 3000
Target Variable: Satisfaction (Satisfied / Not Satisfied)

Total Features: 15 (excluding target)
🔹 Categorical Features (4): 
- Gender
- Customer Type
- Type of Travel
- Flight Class

🔹 Numerical Features (11):
- Age
- Flight Distance
- Departure Delay
- Arrival Delay
- Seat Comfort
- In-flight WiFi Service
- Food and Drink
- Cleanliness
- Baggage Handling
- Check-in Service
- On-board Service


## Approach & Methodology
1. Data Cleaning
    1. Checked missing values and duplicates
    2. Removed duplicates if present
    3. Exploratory Data Analysis (EDA)
    4. Target distribution analysis
    5. Count plots for categorical variables
    6. Understanding class balance
2. Encoding & Feature Scaling
    1. One-Hot Encoding for categorical variables
    2. StandardScaler applied for numerical stability and model convergence
3. Train-Test Split
    1. 80% Training, 20% Testing
4. Base Model Comparison
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Support Vector Machine (SVM)
    4. Decision Tree
    5. Decision Tree selected based on evaluation metrics
5. Ensemble Learning
    1. Random Forest Classifier (Primary Model)
    2. Gradient Boosting Classifier (Experimented)
    3. Hyperparameter Tuning
    4. GridSearchCV used to optimize model performance
6. Dimensionality Reduction
    1. PCA (Principal Component Analysis)
    2. LDA (Linear Discriminant Analysis)
    3. t-SNE (Visualization only)
7. Neural Network Modeling
    1. Artificial Neural Network (ANN) using TensorFlow & Keras
8. Evaluation & Validation
    1. Accuracy, Precision, Recall, F1-score
    2. Cross-validation for robustness
9. real-world Prediction
    1. Tested model with a manually created passenger profile


## Models Used
- Machine Learning
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest Classifier
- Gradient Boosting Classifier
- Deep Learning
- Artificial Neural Network (ANN)


## Dimensionality Reduction
- PCA (Principal Component Analysis)
- LDA (Linear Discriminant Analysis)
- t-SNE (Visualization Only)


## Results
- Random Forest and Gradient Boosting achieved ~94% accuracy
- ANN achieved competitive performance
- Random Forest selected as the final model due to:
- Strong performance
- Stability


## Feature Importance
The Random Forest model revealed that the following features had the highest impact on satisfaction:
- Arrival Delay
- Online Boarding / Service-related features
- Seat Comfort
- Cleanliness


## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow, Keras
- Matplotlib, Seaborn


















