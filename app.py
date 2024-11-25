from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Function to save plot as base64 string
def save_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return base64_img

# Function to preprocess data
def preprocess_data(X, y):
    # Handle categorical features in X
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['number']).columns

    if not categorical_features.empty:
        # Apply one-hot encoding to categorical features
        transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        X = transformer.fit_transform(X)
    else:
        # Scale numerical features if no categorical features exist
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Encode target variable if categorical
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X, y

# Function to evaluate and compare models
def evaluate_models(X, y):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

        # Calculate ROC AUC score if possible
        if is_multiclass:
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro") * 100 if hasattr(model, "predict_proba") else None
        else:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 100 if hasattr(model, "predict_proba") else None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Append results
        results.append({
            "Model": model_name,
            "Accuracy": f"{accuracy:.2f}%",
            "Precision": f"{precision:.2f}%",
            "Recall": f"{recall:.2f}%",
            "F1 Score": f"{f1:.2f}%",
            "AUC": f"{auc:.2f}%" if auc is not None else "N/A"
        })

        # Plot confusion matrix
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
        values = [float(result[metric].strip('%')) for result in results]
        plt.plot(models_list, values, marker='o', label=metric)
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.xticks(rotation=45)
    comparison_chart = save_plot_to_base64()
    plt.close()

    return results, comparison_chart, confusion_matrices

# Route to upload and analyze dataset
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        data = pd.read_csv(file)

        # Separate features and target (assuming the target column is the last one)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Preprocess data
        X, y = preprocess_data(X, y)

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
    app.run(host='0.0.0.0', port=5000, debug=True)
