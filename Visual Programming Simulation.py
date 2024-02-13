import tkinter as tk
from tkinter import ttk
import matplotlib
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from sklearn.model_selection import train_test_split
from sklearn import tree, neural_network, svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model as lm

root = tk.Tk()
root.title("Evaluation Metrics Simulation") #Title of Main Program

# Graph Template
matplotlib.use("TkAgg")

test_data = {
    'Python': 11.27,
    'C': 11.16,
    'Java': 10.46,
    'C++': 7.5,
    'C#': 5.26
}
languages = test_data.keys()
popularity = test_data.values()

figure = Figure(figsize=(4, 4.5), dpi=100)
figure_canvas = FigureCanvasTkAgg(figure, root)

axes = figure.add_subplot()
axes.bar(languages, popularity)
axes.set_title('Top 5 Programming Languages')
axes.set_ylabel('Popularity')
figure_canvas.get_tk_widget().grid(column=2,row=0, sticky=tk.W, padx=5, pady=5, rowspan=25, columnspan=1)

#Center the Program to Screen when opened
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

center_x = int(screen_width/2 - 800 / 2)
center_y = int(screen_height/2 - 600 / 2)

root.geometry(f'800x500+{center_x}+{center_y}')
root.resizable(False, False)

    #Dataset Combo Box
dataset_string = tk.StringVar()
datasets_combo = ttk.Combobox(root, textvar=dataset_string)
datasets_combo['values'] = ('Breast Cancer', 'Diabetes')
datasets_combo['state'] = 'readonly'
datasets_combo.set('Breast Cancer')

    #Data Model Combo Box
datamodel_string = tk.StringVar()
datamodels = ttk.Combobox(root, textvar=datamodel_string)
datamodels['values'] = ('Naive Bayes', 'SVM', 'Decision Tree', 'Neural Network', "Logistic Regression", "Linear Regression")
datamodels['state'] = 'readonly'
datamodels.set('Naive Bayes')

    #Grid Config
root.columnconfigure(2, weight=2)

    # Actual Screen Content
dataset_label = tk.Label(root, text="Pick a sample Dataset:")
dataset_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
datasets_combo.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5, columnspan=2)

datamodel_label = tk.Label(root, text="Pick a sample Data Model:")
datamodel_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
datamodels.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5, columnspan=2)

def load_dataset(dataset_name):
    if dataset_name == 'Breast Cancer':
        return datasets.load_breast_cancer(return_X_y=True), 'Classification'
    elif dataset_name == 'Diabetes':
        return datasets.load_diabetes(return_X_y=True), 'Regression'
    else:
        return None, None

def train_test_model(data_model, dataset, target):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20, train_size=0.80)
    if (data_model == "Naive Bayes"):
        nbayes = GaussianNB().fit(X_train, y_train)
        return nbayes.predict(X_test), y_test
    elif (data_model == "SVM"):
        if(target == "Classification"):
            svectorclass = svm.SVC().fit(X_train, y_train)
            return svectorclass.predict(X_test), y_test
        elif(target == "Regression"):
            svectorreg = svm.SVR().fit(X_train, y_train)
            return svectorreg.predict(X_test), y_test
    elif (data_model == "Decision Tree"):
        treemodel = tree.DecisionTreeRegressor(random_state=0).fit(X_train,y_train)
        return treemodel.predict(X_test), y_test
    elif (data_model == "Neural Network"):
        if(target == "Classification"):
            pony_classifier = neural_network.MLPClassifier().fit(X_train,y_train)
            return pony_classifier.predict(X_test), y_test
        elif(target == "Regression"):
            pony_regressor = neural_network.MLPRegressor().fit(X_train, y_train)
            return pony_regressor.predict(X_test), y_test
    elif (data_model == "Logistic Regression"):
        logreg = lm.LogisticRegression(random_state=0).fit(X_train, y_train)
        return logreg.predict(X_test), y_test
    elif (data_model == "Linear Regression"):
        linreg = lm.LinearRegression().fit(X_train, y_train)
        return linreg.predict(X_test), y_test
    else:
        return None, None

def calculate_metrics():
    dataset_name_var = dataset_string.get()
    dataset, dataset_type_var = load_dataset(dataset_name_var)
    if dataset is None:
        return

    datamodel_name_var = datamodels.get()

    # Update labels for dataset title and data model used
    dataset_name.config(text=f"Dataset Title: {dataset_name_var}")
    datamodel_name.config(text=f"Data Model Used: {datamodel_name_var}")
    dataset_type.config(text=f"Dataset Type: {dataset_type_var}")

    # Sample evaluation metrics
    y_pred, y_true = train_test_model(datamodel_name_var, dataset, dataset_type_var)

    #TODO: Prediction Codes

    if dataset_type_var == 'Classification':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)

        # Update labels for classification metrics
        accuracy_label.grid()
        precision_label.grid()
        recall_label.grid()
        f1_label.grid()
        mcc_label.grid()
        accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
        precision_label.config(text=f"Precision: {precision:.2f}")
        recall_label.config(text=f"Recall: {recall:.2f}")
        f1_label.config(text=f"F1 Score: {f1:.2f}")
        mcc_label.config(text=f"Matthews Correlation Coefficient: {mcc:.2f}")

        # Clear labels for regression metrics
        mae_label.grid_remove()
        mse_label.grid_remove()
        rmse_label.grid_remove()
        mpe_label.grid_remove()
        r2_label.grid_remove()

    elif dataset_type_var == 'Regression':
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mpe = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Clear labels for classification metrics
        accuracy_label.grid_remove()
        precision_label.grid_remove()
        recall_label.grid_remove()
        f1_label.grid_remove()
        mcc_label.grid_remove()

        # Update labels for regression metrics
        mae_label.grid()
        mse_label.grid()
        rmse_label.grid()
        mpe_label.grid()
        r2_label.grid()
        mae_label.config(text=f"Mean Absolute Error: {mae:.2f}")
        mse_label.config(text=f"Mean Squared Error: {mse:.2f}")
        rmse_label.config(text=f"Root Mean Squared Error: {rmse:.2f}")
        mpe_label.config(text=f"Mean Absolute Percentage Error: {mpe:.2f}")
        r2_label.config(text=f"R2 Score: {r2:.2f}")

# Button to calculate metrics
calculate_button = ttk.Button(root, text="Calculate Metrics", command=calculate_metrics)
calculate_button.grid(column=0, row=4, padx=5, pady=5)

tk.Label(root, text="").grid(column=0, row=5, padx=5, pady=5)

evalmetrics_label = tk.Label(root, text="Evaluation Metrics")
evalmetrics_label.grid(column=0, row=6, padx=5, pady=5)

# Labels for displaying classification metrics
accuracy_label = tk.Label(root, text=f"Accuracy: ")
accuracy_label.grid(column=1, row=8, padx=5, pady=5)

precision_label = tk.Label(root, text=f"Precision: ")
precision_label.grid(column=0, row=8, padx=5, pady=5)

recall_label = tk.Label(root, text=f"Recall: ")
recall_label.grid(column=0, row=9, padx=5, pady=5)

f1_label = tk.Label(root, text=f"F1 Score: ")
f1_label.grid(column=1, row=9, padx=5, pady=5)

mcc_label = tk.Label(root, text=f"Matthews Correlation Coefficient: ")
mcc_label.grid(column=0, row=10, padx=5, pady=5, columnspan=2)

# Labels for displaying regression metrics, starts out Invisible
mae_label = tk.Label(root, text="")
mae_label.grid(column=0, row=8, padx=5, pady=5, columnspan=2)
mae_label.grid_remove()

mse_label = tk.Label(root, text="")
mse_label.grid(column=0, row=9, padx=5, pady=5, columnspan=2)
mse_label.grid_remove()

rmse_label = tk.Label(root, text="")
rmse_label.grid(column=0, row=10, padx=5, pady=5, columnspan=2)
rmse_label.grid_remove()

mpe_label = tk.Label(root, text="")
mpe_label.grid(column=0, row=11, padx=5, pady=5, columnspan=2)
mpe_label.grid_remove()

r2_label = tk.Label(root, text="")
r2_label.grid(column=0, row=12, padx=5, pady=5, columnspan=2)
r2_label.grid_remove()

dataset_name_var = "Breast Cancer"
dataset_name = tk.Label(root, text=f"Dataset Title: {dataset_name_var}")
dataset_name.grid(column=1, row=1, sticky=tk.EW, padx=5, pady=5)

dataset_type_var = "Classification" #Some function to determine Target type here
dataset_type = tk.Label(root, text=f"Dataset Type: {dataset_type_var}")
dataset_type.grid(column=1, row=2, sticky=tk.EW, padx=5, pady=5)

datamodel_name_var = "Naive Bayes"
datamodel_name = tk.Label(root, text=f"Data Model Used: {datamodel_name_var}")
datamodel_name.grid(column=1, row=3, sticky=tk.EW, padx=5, pady=5)

# keep the window displaying
root.mainloop()