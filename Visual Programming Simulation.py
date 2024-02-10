import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Evaluation Metrics Simulation") #Title of Main Program

#Center the Program to Screen when opened
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

center_x = int(screen_width/2 - 800 / 2)
center_y = int(screen_height/2 - 600 / 2)

root.geometry(f'800x600+{center_x}+{center_y}')
root.resizable(False, False)

    #Dataset Combo Box
dataset_string = tk.StringVar()
datasets = ttk.Combobox(root, textvariable=dataset_string)
datasets['values'] = ('Dataset 1', 'Dataset 2', 'Dataset 3')
datasets['state'] = 'readonly'
# datasets.bind('<<ComboboxSelected>>', callback)
datasets.set('Dataset 1')

    #Data Model Combo Box
datamodel_string = tk.StringVar()
datamodels = ttk.Combobox(root, textvar=datamodel_string)
datamodels['values'] = ('Data Model 1', 'Data Model 2', 'Data Model 3')
datamodels['state'] = 'readonly'
# datamodels.bind('<<ComboboxSelected>>', callback)
datamodels.set('Data Model 1')

    #Grid Config
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=1)

    # Actual Screen Content
dataset_label = tk.Label(root, text="Pick a sample Dataset:")
dataset_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
datasets.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

datamodel_label = tk.Label(root, text="Pick a sample Data Model:")
datamodel_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
datamodels.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

evalmetrics_label = tk.Label(root, text="Evaluation Metrics")
evalmetrics_label.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)

# keep the window displaying
root.mainloop()