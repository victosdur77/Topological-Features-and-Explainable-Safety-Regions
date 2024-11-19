import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from skrules import SkopeRules # in skope_rules.py, from sklearn.externals import six must be changed with import six

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import joblib
import matplotlib.pyplot as plt



def OptimizedSkopeRules(X_train, y_train, param_grid, cvmetric = "accuracy",save_rules=True,rules_path="./best_skope_rules.csv", save_model = True, save_path = "./best_svm.sav"):
    
    ''' Train a SkopeRules model with RandomizedSearch 5-fold Cross-Validation  over the parameters defined by the param_grid. The optimal solution is found based on the cvmetric (default is "accuracy". Returns the trained model on the whole train data, and saves it to file. X_train and y_train are pandas dataframes'''

    skope_model = SkopeRules(feature_names = X_train.columns, random_state=12)

    t_start=time.time()
    tunedmodel_random = RandomizedSearchCV(skope_model, param_grid, cv=5, scoring=cvmetric,return_train_score=True, random_state = 25) 
    tunedmodel_random = tunedmodel_random.fit(X_train, y_train)
    t_end = time.time()
    t_random = t_end - t_start
    print("Time spent for Randomized Search: ", t_random, " s")

    resrandom = tunedmodel_random.cv_results_
    for acc,par in zip(resrandom["mean_test_score"],resrandom["params"]):
        print(np.mean(acc), par)

    # get the best parameters 
    bestparams = tunedmodel_random.best_params_
    print("SkopeRules best parameters with Randomized Search:")
    print(bestparams)
    
    bestmodel = SkopeRules(feature_names = X_train.columns, random_state=12)
    bestmodel.set_params(**bestparams)
    bestmodel = bestmodel.fit(X_train, y_train)
    
    if save_model:
        joblib.dump(bestmodel, save_path)
    

    # save obtained rules
    rules = bestmodel.rules_

        #filename_rules = "simulation2/skope/skope_rules_collisions.csv"
    class_label = y_train.name
    count = 1

    if "noncollisions" in rules_path:
        label = str(0)
    else:
        label = str(1)
    for rule in rules:
        print("RULE {}: IF ".format(count)+str(rule[0])+" THEN "+class_label+" = " + label + ", COVERING: {}, ERROR: {}".format(rule[1][1],1-rule[1][0])+"\n")
        if save_rules:
            with open(rules_path,"a") as rulefile:
                rulefile.write("RULE {}: IF ".format(count)+str(rule[0])+" THEN "+class_label+" = " + label + ", COVERING: {}, ERROR: {}".format(rule[1][1],1-rule[1][0])+"\n")
        count+=1
    
    
    return bestmodel