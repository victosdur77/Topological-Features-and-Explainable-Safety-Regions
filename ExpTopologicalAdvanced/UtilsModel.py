import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, f1_score, recall_score
import time



def EvaluateModel(bestmodel, X_test, y_test, modelname):
    
    ''' Given a trained binary classification model, shows the confusion matrix and computes some metrics on test data'''
    
    # test
    y_svm = bestmodel.predict(X_test).reshape((len(X_test),1))
    #print(y_SK)



    #print(list(zip(list(y_test),list(y_SK))))
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
    print("ACC = {}, F1 = {}, PPV = {}, NPV = {}, TPR = {}, TNR = {}\n".format(accuracy,f1,PPV,NPV,TPR,TNR))
    print("TP = {}, FP = {}, TN = {}, FN = {}".format(TP,FP,TN,FN))
