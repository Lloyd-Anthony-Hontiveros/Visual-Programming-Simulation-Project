import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

root = tk.Tk()
root.title("Evaluation Metrics Simulation")

bg_color = "#f3f3f3"
text_color = "#000000"
button_bg_color = "#c5c5c5"

root.configure(bg=bg_color)

# Function to load dataset
def load_dataset(dataset_name):
    if dataset_name == 'Iris':
        return datasets.load_iris(), 'classification'
    elif dataset_name == 'Diabetes':
        return datasets.load_diabetes(), 'regression'
    else:
        return None, None

# Function to calculate metrics
def calculate_metrics():
    dataset_name_var = dataset_string.get()
    dataset, problem_type = load_dataset(dataset_name_var)
    if dataset is None:
        return

    # Update labels for dataset title and data model used
    dataset_title_label.config(text=f"Dataset Title: {dataset_name_var}", fg=text_color)
    datamodel_used_label.config(text=f"Data Model Used: {'Decision Tree' if problem_type == 'classification' else 'Linear Regression'}", fg=text_color)

    # Update evaluation metrics type label
    evalmetrics_type_label.config(text=f"Evaluation Metrics Type: {problem_type.capitalize()}", fg=text_color)

    # Sample evaluation metrics
    y_true = dataset.target
    y_pred = np.random.randint(0, high=np.max(y_true), size=len(y_true))

    if problem_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        accuracy_label.config(text=f"Accuracy: {accuracy:.2f}", fg=text_color)
        precision_label.config(text=f"Precision: {precision:.2f}", fg=text_color)
        recall_label.config(text=f"Recall: {recall:.2f}", fg=text_color)
        f1_label.config(text=f"F1 Score: {f1:.2f}", fg=text_color)

        mse_label.config(text="", fg=text_color)
        rmse_label.config(text="", fg=text_color)
        mae_label.config(text="", fg=text_color)
        r2_label.config(text="", fg=text_color)
        
        # Display data visualization for decision tree
        plot_decision_tree(dataset)

    elif problem_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        accuracy_label.config(text="", fg=text_color)
        precision_label.config(text="", fg=text_color)
        recall_label.config(text="", fg=text_color)
        f1_label.config(text="", fg=text_color)

        mse_label.config(text=f"Mean Squared Error: {mse:.2f}", fg=text_color)
        rmse_label.config(text=f"Root Mean Squared Error: {rmse:.2f}", fg=text_color)
        mae_label.config(text=f"Mean Absolute Error: {mae:.2f}", fg=text_color)
        r2_label.config(text=f"R2 Score: {r2:.2f}", fg=text_color)
        
        # Display data visualization for linear regression
        plot_linear_regression(dataset)

# Function to plot decision tree
def plot_decision_tree(dataset):
    X = dataset.data
    y = dataset.target

    model = DecisionTreeClassifier()
    model.fit(X, y)

    plt.figure(figsize=(10, 7))
    plot_tree(model, filled=True, feature_names=dataset.feature_names, class_names=dataset.target_names)
    plt.title('Decision Tree Visualization')
    plt.show()

# Function to plot linear regression
def plot_linear_regression(dataset):
    X = dataset.data[:, np.newaxis, 2]
    y = dataset.target

    model = LinearRegression()
    model.fit(X, y)

    plt.scatter(X, y, color='black')
    plt.plot(X, model.predict(X), color='blue', linewidth=3)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression')
    plt.show()

# Function to clear all labels
def clear_all():
    dataset_combo.set('Sample datasets')
    dataset_title_label.config(text="", fg=text_color)
    datamodel_used_label.config(text="", fg=text_color)
    evalmetrics_type_label.config(text="", fg=text_color)
    accuracy_label.config(text="", fg=text_color)
    precision_label.config(text="", fg=text_color)
    recall_label.config(text="", fg=text_color)
    f1_label.config(text="", fg=text_color)
    mse_label.config(text="", fg=text_color)
    rmse_label.config(text="", fg=text_color)
    mae_label.config(text="", fg=text_color)
    r2_label.config(text="", fg=text_color)

# Dataset selection widgets
dataset_label = tk.Label(root, text="Pick a sample Dataset:", bg=bg_color, fg=text_color)
dataset_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

dataset_string = tk.StringVar()
dataset_combo = ttk.Combobox(root, textvariable=dataset_string)
dataset_combo['values'] = ('Iris', 'Diabetes')
dataset_combo['state'] = 'readonly'
dataset_combo.set('Sample datasets')
dataset_combo.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

# Evaluation metrics label
evalmetrics_label = tk.Label(root, text="Evaluation Metrics", bg=bg_color, fg=text_color)
evalmetrics_label.grid(column=0, row=6, sticky=tk.W, padx=5, pady=5)

# Calculate Metrics button
calculate_button = ttk.Button(root, text="Calculate Metrics", command=calculate_metrics, style='My.TButton')
calculate_button.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)

# Clear All button
clear_button = ttk.Button(root, text="Clear All", command=clear_all, style='My.TButton')
clear_button.grid(column=1, row=7, sticky=tk.W, padx=5, pady=5)

# Labels for displaying information
dataset_title_label = tk.Label(root, text="", bg=bg_color, fg=text_color)
dataset_title_label.grid(column=0, row=8, sticky=tk.W, padx=5, pady=5)

evalmetrics_type_label = tk.Label(root, text="", bg=bg_color, fg=text_color)
evalmetrics_type_label.grid(column=0, row=9, sticky=tk.W, padx=5, pady=5)

datamodel_used_label = tk.Label(root, text="", bg=bg_color, fg=text_color)
datamodel_used_label.grid(column=0, row=10, sticky=tk.W, padx=5, pady=5)

accuracy_label = tk.Label(root, text="Accuracy: ", bg=bg_color, fg=text_color)
accuracy_label.grid(column=1, row=8, sticky=tk.W, padx=5, pady=5)

precision_label = tk.Label(root, text="Precision: ", bg=bg_color, fg=text_color)
precision_label.grid(column=1, row=9, sticky=tk.W, padx=5, pady=5)

recall_label = tk.Label(root, text="Recall: ", bg=bg_color, fg=text_color)
recall_label.grid(column=1, row=10, sticky=tk.W, padx=5, pady=5)

f1_label = tk.Label(root, text="F1 Score: ", bg=bg_color, fg=text_color)
f1_label.grid(column=1, row=11, sticky=tk.W, padx=5, pady=5)

mse_label = tk.Label(root, text="Mean Squared Error: ", bg=bg_color, fg=text_color)
mse_label.grid(column=1, row=12, sticky=tk.W, padx=5, pady=5)

rmse_label = tk.Label(root, text="Root Mean Squared Error: ", bg=bg_color, fg=text_color)
rmse_label.grid(column=1, row=13, sticky=tk.W, padx=5, pady=5)

mae_label = tk.Label(root, text="Mean Absolute Error: ", bg=bg_color, fg=text_color)
mae_label.grid(column=1, row=14, sticky=tk.W, padx=5, pady=5)

r2_label = tk.Label(root, text="R2 Score: ", bg=bg_color, fg=text_color)
r2_label.grid(column=1, row=15, sticky=tk.W, padx=5, pady=5)

style = ttk.Style()
style.configure('My.TButton', background=button_bg_color)

# Bind dataset combo box selection event
def update_data_models(event):
    pass

dataset_combo.bind("<<ComboboxSelected>>", update_data_models)

root.mainloop()
