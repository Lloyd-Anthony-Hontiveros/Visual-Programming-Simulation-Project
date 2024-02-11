import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

root = tk.Tk()
root.title("Evaluation Metrics Simulation") #Title of Main Program

#Center the Program to Screen when opened
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

center_x = int(screen_width/2 - 800 / 2)
center_y = int(screen_height/2 - 600 / 2)

root.geometry(f'800x600+{center_x}+{center_y}')
root.resizable(False, False)

# Dataset Combo Box
dataset_string = tk.StringVar()
dataset_combo = ttk.Combobox(root, textvariable=dataset_string)
dataset_combo['values'] = ('Iris', 'Digits', 'Wine', 'Breast Cancer', 'Diabetes', 'Boston Housing', 'California Housing', 'Ames Housing')
dataset_combo['state'] = 'readonly'
dataset_combo.set('Iris')

# Data Model Combo Box
datamodel_string = tk.StringVar()
datamodels = ttk.Combobox(root, textvar=datamodel_string)
datamodels['values'] = ('Logistic Regression', 'Random Forest', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'Gradient Boosting')
datamodels['state'] = 'readonly'
datamodels.set('Logistic Regression')


# Grid Config
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

# Actual Screen Content
dataset_label = tk.Label(root, text="Pick a sample Dataset:")
dataset_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
dataset_combo.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

datamodel_label = tk.Label(root, text="Pick a sample Data Model:")
datamodel_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
datamodels.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

evalmetrics_label = tk.Label(root, text="Evaluation Metrics")
evalmetrics_label.grid(column=0, row=6, sticky=tk.W, padx=5, pady=5)

# Define a function to load datasets based on the selected name
def load_dataset(dataset_name):
    if dataset_name == 'Iris':
        return datasets.load_iris(), 'classification'
    elif dataset_name == 'Digits':
        return datasets.load_digits(), 'classification'
    elif dataset_name == 'Wine':
        return datasets.load_wine(), 'classification'
    elif dataset_name == 'Breast Cancer':
        return datasets.load_breast_cancer(), 'classification'
    elif dataset_name == 'Diabetes':
        return datasets.load_diabetes(), 'regression'
    elif dataset_name == 'Boston Housing':
        return datasets.fetch_openml(name="boston"), 'regression'
    elif dataset_name == 'California Housing':
        return datasets.fetch_california_housing(), 'regression'
    elif dataset_name == 'Ames Housing':
        return datasets.fetch_openml(name="house_prices", as_frame=True), 'regression'
    else:
        return None, None

def calculate_metrics():
    dataset_name_var = dataset_string.get()
    dataset, problem_type = load_dataset(dataset_name_var)
    if dataset is None:
        return

    datamodel_name_var = datamodels.get()

    # Update labels for dataset title and data model used
    dataset_title_label.config(text=f"Dataset Title: {dataset_name_var}")
    datamodel_used_label.config(text=f"Data Model Used: {datamodel_name_var}")

    # Sample evaluation metrics
    y_true = dataset.target
    y_pred = np.random.randint(0, high=np.max(y_true), size=len(y_true))


    if problem_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Update labels for classification metrics
        accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
        precision_label.config(text=f"Precision: {precision:.2f}")
        recall_label.config(text=f"Recall: {recall:.2f}")
        f1_label.config(text=f"F1 Score: {f1:.2f}")

        # Clear labels for regression metrics
        mse_label.config(text="")
        r2_label.config(text="")

    elif problem_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Clear labels for classification metrics
        accuracy_label.config(text="")
        precision_label.config(text="")
        recall_label.config(text="")
        f1_label.config(text="")

        # Update labels for regression metrics
        mse_label.config(text=f"Mean Squared Error: {mse:.2f}")
        r2_label.config(text=f"R2 Score: {r2:.2f}")

# Button to calculate metrics
calculate_button = ttk.Button(root, text="Calculate Metrics", command=calculate_metrics)
calculate_button.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)

# Define the label for dataset title
dataset_title_label = tk.Label(root, text="")
dataset_title_label.grid(column=0, row=8, sticky=tk.W, padx=5, pady=5)

# Define the label for data model used
datamodel_used_label = tk.Label(root, text="")
datamodel_used_label.grid(column=0, row=9, sticky=tk.W, padx=5, pady=5)


# Labels for displaying classification metrics
accuracy_label = tk.Label(root, text="Accuracy: ")
accuracy_label.grid(column=1, row=8, sticky=tk.W, padx=5, pady=5)

precision_label = tk.Label(root, text="Precision: ")
precision_label.grid(column=1, row=9, sticky=tk.W, padx=5, pady=5)

recall_label = tk.Label(root, text="Recall: ")
recall_label.grid(column=1, row=10, sticky=tk.W, padx=5, pady=5)

f1_label = tk.Label(root, text="F1 Score: ")
f1_label.grid(column=1, row=11, sticky=tk.W, padx=5, pady=5)

# Labels for displaying regression metrics
mse_label = tk.Label(root, text="Mean Squared Error: ")
mse_label.grid(column=1, row=12, sticky=tk.W, padx=5, pady=5)

r2_label = tk.Label(root, text="R2 Score: ")
r2_label.grid(column=1, row=13, sticky=tk.W, padx=5, pady=5)

# Keep the window displaying
root.mainloop()
