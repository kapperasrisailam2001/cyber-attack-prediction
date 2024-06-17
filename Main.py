import tkinter as tk
from tkinter import messagebox, filedialog, Text, Scrollbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import os

# Global variables
filename = None
le = None
X, Y = None, None
dataset = None
accuracy = []
precision = []
recall = []
fscore = []
X_train, X_test, y_train, y_test = None, None, None, None
classifier = None
labels = ['Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS', 'Infilteration', 'SQL Injection']

# Create main window
main = tk.Tk()
main.title("A Data Analytics Approach to the Cybercrime Underground Economy")
main.geometry("1300x1200")
main.config(bg='LightSkyBlue')

# Functions
def uploadDataset():
    global filename, dataset
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, f"{filename} loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    text.insert(tk.END, "Dataset before preprocessing\n\n")
    text.insert(tk.END, str(dataset.head()) + "\n\n")
    unique, count = np.unique(dataset['Label'], return_counts=True)
    for i in range(len(unique)):
        text.insert(tk.END, f"{unique[i]} = {count[i]}\n")
    text.update_idletasks()
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.xlabel('Various Cybercrime Attacks')
    plt.ylabel('Count')
    plt.title('Cybercrime Graph')
    plt.show()

def analyticalProcessing():
    global X, Y, le, dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', tk.END)
    dataset.drop(columns=['Timestamp', 'Flow Byts/s', 'Flow Pkts/s'], inplace=True)
    le = LabelEncoder()
    dataset['Label'] = le.fit_transform(dataset['Label'].astype(str))
    dataset = dataset.values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    sm = SMOTE(random_state=42)
    smote_X, smote_Y = sm.fit_resample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(smote_X, smote_Y, test_size=0.2, random_state=0)
    text.insert(tk.END, "Dataset after Preprocessing\n\n")
    text.insert(tk.END, str(X) + "\n\n")
    text.insert(tk.END, f"Total records found in dataset: {X.shape[0]}\n")
    text.insert(tk.END, f"Total features found in dataset: {X.shape[1]}\n\n")
    text.insert(tk.END, "Dataset Train and Test Split\n\n")
    text.insert(tk.END, f"80% dataset records used to train Naive Bayes algorithms: {X_train.shape[0]}\n")
    text.insert(tk.END, f"20% dataset records used to train Naive Bayes algorithms: {X_test.shape[0]}\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(tk.END, f"{algorithm} Accuracy  :  {a}\n")
    text.insert(tk.END, f"{algorithm} Precision : {p}\n")
    text.insert(tk.END, f"{algorithm} Recall    : {r}\n")
    text.insert(tk.END, f"{algorithm} FScore    : {f}\n\n")

def runNaiveBayes():
    global X, Y, X_train, X_test, y_train, y_test, classifier
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', tk.END)
    model_path = 'model/nb.txt'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            classifier = pickle.load(file)
    else:
        classifier = GaussianNB()
        classifier.fit(X, Y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(classifier, file)
    predict = classifier.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)

def predict():
    global classifier
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, f"{filename} loaded\n\n")
    dataset = pd.read_csv(filename, encoding='iso-8859-1')
    dataset.fillna(0, inplace=True)
    dataset.drop(columns=['Timestamp', 'Flow Byts/s', 'Flow Pkts/s'], inplace=True)
    dataset = dataset.values
    prediction = classifier.predict(dataset)
    for i in range(len(prediction)):
        text.insert(tk.END, f"Test DATA : {dataset[i]} ===> PREDICTED AS {labels[int(prediction[i])]}\n\n")

def graph():
    df = pd.DataFrame([
        ['Naive Bayes', 'Precision', precision[0]],
        ['Naive Bayes', 'Recall', recall[0]],
        ['Naive Bayes', 'F1 Score', fscore[0]],
        ['Naive Bayes', 'Accuracy', accuracy[0]],
    ], columns=['Algorithms', 'Performance Output', 'Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def close():
    main.destroy()

# UI Elements
font = ('times', 16, 'bold')
title = tk.Label(main, text='A Data Analytics Approach to the Cybercrime Underground Economy', bg='greenyellow', fg='dodger blue', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Dataset Upload & Analysis", command=uploadDataset, font=font1)
uploadButton.place(x=50, y=550)

processButton = tk.Button(main, text="Dataset Processing & Analytical Methods", command=analyticalProcessing, font=font1)
processButton.place(x=370, y=550)

nbButton = tk.Button(main, text="Run Naive Bayes Classification Model", command=runNaiveBayes, font=font1)
nbButton.place(x=750, y=550)

graphButton = tk.Button(main, text="Classification Performance Graph", command=graph, font=font1)
graphButton.place(x=50, y=600)

predictButton = tk.Button(main, text="Predict Cyber Crime", command=predict, font=font1)
predictButton.place(x=370, y=600)

main.mainloop()
