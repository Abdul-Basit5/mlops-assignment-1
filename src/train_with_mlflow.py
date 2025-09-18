import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set tracking URI (local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
mlflow.set_experiment("iris_experiment")

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# ✅ Add noise (taake accuracy perfect 1.0 na aaye)
rng = np.random.RandomState(42)
noise = rng.normal(0, 0.3, X.shape)   # thoda kam noise rakha
X = X + noise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42),
    "SVM": SVC(kernel="linear", C=0.5, random_state=42)
}

# Example input (ek row sample data)
input_example = np.array([X_train[0]])

# Run MLflow logging
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log parameters & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Save model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=name.replace(" ", "_"),
            input_example=input_example,
            registered_model_name=f"{name.replace(' ', '_')}_Model"
        )

print("✅ Training finished. Run 'mlflow ui' and check results")
