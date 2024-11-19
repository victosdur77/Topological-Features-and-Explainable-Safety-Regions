import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score



def EvaluateModel(bestmodel, X_test, y_test, modelname, showcm = True):
    
    ''' Given a trained binary classification model, shows the confusion matrix and computes some metrics on test data'''
    
    # test
    y_svm = bestmodel.predict(X_test) 

    if showcm:
        cm_svm = confusion_matrix(y_test, y_svm)
        cmSVM = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels = bestmodel.classes_)
        cmSVM.plot()
        cmSVM.ax_.set_title("{}".format(modelname))
        plt.show()

    TN, FP, FN, TP = confusion_matrix(y_test, y_svm).ravel()

    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = (2*TP)/(2*TP+FP+FN)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(TN+FP)
    FNR = FN/(TP+FN)
    print("ACC = {}, F1 = {}, PPV = {}, NPV = {}, TPR = {}, TNR = {}, FPR = {}, FNR = {}\n".format(accuracy,f1,PPV,NPV,TPR,TNR,FPR,FNR))
    print("TP = {}, FP = {}, TN = {}, FN = {}".format(TP,FP,TN,FN))

    metrics = {"TP":TP,"FP":FP,"TN":TN,"FN":FN, "accuracy": accuracy,"F1": f1,"PPV": PPV,"NPV": NPV,"TPR":TPR,"TNR":TNR,"FPR":FPR,"FNR":FNR}
    return metrics

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==-1:
           TN += 1
        if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
           FN += 1
        
    P = TP + FN
    N = TN + FP

    return({'TPR':TP/P, 'FPR': FP/N, 'TNR': TN/N, 'FNR': FN/P})