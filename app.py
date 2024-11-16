import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for Flask

import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import base64
import io

app = Flask(__name__)

# Function to save plot as base64 string
def save_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return base64_img

# Function to evaluate and compare models
def evaluate_models(X, y):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),  # Enable probability for ROC-AUC
        "KNN": KNeighborsClassifier()
    }

    # Store results and confusion matrices
    results = []
    confusion_matrices = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        # Check if this is a binary or multiclass problem
        is_multiclass = len(np.unique(y)) > 2

        # Calculate metrics as floats
        accuracy = float(accuracy_score(y_test, y_pred) * 100)
        precision = float(precision_score(y_test, y_pred, average='macro', zero_division=0) * 100)
        recall = float(recall_score(y_test, y_pred, average='macro', zero_division=0) * 100)
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0) * 100)

        # Calculate ROC AUC score if it's a binary problem
        if is_multiclass:
            try:
                auc = float(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro") * 100)
            except AttributeError:
                auc = None  # Some models do not support `predict_proba` in multiclass setting
        else:
            auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 100) if hasattr(model, "predict_proba") else None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Append results with raw float values (percentage format will be applied later)
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc if auc is not None else "N/A"
        })

        # Plot and save confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        confusion_matrices.append({
            "name": model_name,
            "plot": save_plot_to_base64()
        })
        plt.close()

    # Generate line plot for metric comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models_list = [result["Model"] for result in results]

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        values = [result[metric] for result in results if isinstance(result[metric], (int, float))]
        plt.plot(models_list, values, marker='o', label=metric)
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.xticks(rotation=45)
    comparison_chart = save_plot_to_base64()
    plt.close()

    # Convert results to display-friendly format (as strings with % symbols)
    formatted_results = []
    for result in results:
        formatted_results.append({
            "Model": result["Model"],
            "Accuracy": f"{result['Accuracy']:.2f}%" if isinstance(result['Accuracy'], (int, float)) else result['Accuracy'],
            "Precision": f"{result['Precision']:.2f}%" if isinstance(result['Precision'], (int, float)) else result['Precision'],
            "Recall": f"{result['Recall']:.2f}%" if isinstance(result['Recall'], (int, float)) else result['Recall'],
            "F1 Score": f"{result['F1 Score']:.2f}%" if isinstance(result['F1 Score'], (int, float)) else result['F1 Score'],
            "AUC": f"{result['AUC']:.2f}%" if isinstance(result['AUC'], (int, float)) else result['AUC']
        })

    return formatted_results, comparison_chart, confusion_matrices

# Route to upload and analyze dataset
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        data = pd.read_csv(file)

        # Separate features and target (assuming the target column is the last one)
        X = data.iloc[:, :-1]    # Features
        y = data.iloc[:, -1]     # Target

        # Evaluate models and get results
        results, comparison_chart, confusion_matrices = evaluate_models(X, y)

        # Pass results to the HTML template
        return render_template(
            'index.html',
            results=results,
            comparison_chart=comparison_chart,
            confusion_matrices=confusion_matrices
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=80000)
