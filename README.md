# MLOps Assignment 1: Model Training, Tracking with MLflow, and GitHub Basics

## 🎯 Objective
The purpose of this assignment is to gain hands-on experience with:
1. Using **GitHub** for version control and collaboration.  
2. Building and training **multiple ML models** for comparison.  
3. Using **MLflow** for experiment tracking, logging, monitoring, and model registration.  
4. Ensuring all code and results are **structured, reproducible, and documented**.  

---

## 📝 Problem Statement
We aim to train and compare multiple machine learning models on a classification dataset,  
track their performance using **MLflow**, and register the best model for deployment.

---

## 📊 Dataset
- **Dataset Used:** Iris dataset (from `sklearn.datasets`)  
- **Features:** 4 numerical features (Sepal length, Sepal width, Petal length, Petal width)  
- **Target:** 3 classes (Setosa, Versicolor, Virginica)  
- **Size:** 150 samples  

---

## Models & Results
| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.90     | 0.90      | 0.90   | 0.90     |
| Random Forest         | 0.90     | 0.90      | 0.90   | 0.90     |
| SVM                   | 0.86     | 0.86      | 0.86   | 0.86     |

✅ Best Model: **Logistic Regression**

---

## MLflow
- Logged metrics, parameters, and confusion matrix.  
- Compared multiple runs in MLflow UI.  
- Registered Logistic Regression as **version 1**.  

---

## Run Project
```bash
git clone https://github.com/Abdul-Basit5/mlops-assignment-1.git
cd mlops-assignment-1
pip install -r requirements.txt
mlflow ui
python notebooks/train_models.py
