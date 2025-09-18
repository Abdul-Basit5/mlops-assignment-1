# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# ✅ Add noise to dataset to reduce model performance
rng = np.random.RandomState(42)
noise = rng.normal(0, 0.5, X.shape)   # random noise
X = X + noise

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3),  # limit depth to reduce accuracy
    "SVM": SVC()
}

results = {}

# Step 5: Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    # Save model
    joblib.dump(model, f"../models/{name.replace(' ', '_')}.pkl")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"../results/{name.replace(' ', '_')}_cm.png")
    plt.close()

# Step 6: Show comparison
df_results = pd.DataFrame(results).T
print(df_results)
