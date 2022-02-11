from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import pickle

from IterativeSMOSVM import SVM #importing iterative SVM-SMO object as SVM

main = Tk()
main.title("Machine Learning Inspired Codeword Selection for Dual Connectivity in 5G User-centric Ultra-dense Networks")
main.geometry("1300x1200")

global X, Y
asr = []
global X_train, X_test, y_train, y_test
global dataset

def uploadDataset():
    global dataset
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")

    dataset = pd.read_csv(filename,nrows=500)
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('Class').size()
    label.plot(kind="bar")
    plt.title("Total Codewords for Beam Selection available in dataset")
    plt.show()
    
def calculateASR(y, y_hat):
    asr = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            asr += 1
    return float(asr)/len(y)

def preprocess():
    global X, Y
    global asr
    global X_train, X_test, y_train, y_test
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total Base Station signal records found in dataset: "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each base station signal: "+str(X.shape[1])+"\n\n")
    text.insert(END,"Total records used to train Iterative SVM-SMO Algorithm: "+str(X_train.shape[0])+"\n")
    

def runASR():
    text.delete('1.0', END)
    global X, Y
    global asr
    global X_train, X_test, y_train, y_test
    asr.clear()
    svm_cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    asr_rate = calculateASR(y_test, predict)
    asr.append(asr_rate)
    text.insert(END,"ASR Based Algorithm Correct Prediction Rate: "+str(asr_rate)+"\n\n")

def proposeISIC():
    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            model = pickle.load(file)
        file.close()
    else:
        #creating SVM object with max iteration as 10 and then calling fit function to train svm on X and Y train data
        model = SVM(max_iter=10, kernel_type='linear', C=1.0, epsilon=0.001)
        model.fit(X_train, y_train)
    #performing code words classification on trained model using test data to calculate sum of correct classification result     
    predict = model.predict(X_test)
    asr_rate = calculateASR(y_test, predict)
    asr.append(asr_rate)
    text.insert(END,"Propose Iterative SVM-SMO Classification (ISSC) ASR: "+str(asr_rate)+"\n\n")

def graph():
    global asr
    height = asr
    bars = ('Average Sum Rate (ASR)','Propose ISSC ASR')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("ASR Based Algorithm VS Propose ISSC ASR")
    plt.show()


def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Machine Learning Inspired Codeword Selection for Dual Connectivity in 5G User-centric Ultra-dense Networks')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload 5G Beam Selection Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

asrButton = Button(main, text="Run Average Sum Rate Algorithm", command=runASR)
asrButton.place(x=20,y=200)
asrButton.config(font=ff)

svmsmoButton = Button(main, text="Run Propose ISSC Algorithm", command=proposeISIC)
svmsmoButton.place(x=20,y=250)
svmsmoButton.config(font=ff)

graphButton = Button(main, text="ASR Comparison Graph", command=graph)
graphButton.place(x=20,y=300)
graphButton.config(font=ff)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=20,y=350)
predictButton.config(font=ff)




font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
