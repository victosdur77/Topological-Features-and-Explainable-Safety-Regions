import os
from os.path import exists

import anchor
import anchor.anchor_tabular
import pandas as pd
import numpy as np

import joblib

import time

from sklearn.inspection import DecisionBoundaryDisplay

import operator as op

import matplotlib.pyplot as plt


operators = {'<': op.lt,'<=': op.le,'=': op.eq,'>': op.gt,'>=': op.ge}


def GenerateAnchorRules(bestmodel,X_train, X_test, y_test, yt, test_indexes,feature_labels, class_names, precision_threshold = 0.95, filepath = "anchors.csv"):
    
    ''' Given a trained ML model, individuates local rules for test set points. (Info here: https://christophm.github.io/interpretable-ml-book/anchors.html )
    These rules are saved to a CSV file and a list of the explanation objects is also returned '''
    
    explainer = anchor.anchor_tabular.AnchorTabularExplainer(class_names,feature_labels,X_train)

    if not exists(filepath):
        with open(filepath,'a') as output_file:
               output_file.write('Index'+','+'AnchorConditions'+','+'Coverage'+','+'Precision'+','+'AnchorOutput'+','+'ModelOutput'+','+'RealOutput'+'\n')
    #cover=np.empty(len(X_test_t))
    explanations=[]
    t_start = time.time()
    for i in range(0,len(X_test)):
            #print(X_test[i])
            exp=explainer.explain_instance(X_test[i], bestmodel.predict, threshold = precision_threshold)
            explanations.append(exp)

            conditions_exp = exp.names()

            if conditions_exp == []:
                continue
            else:
                premise = conditions_exp[0]
                for s in conditions_exp[1:]:
                    premise = premise +" AND "+s     
                rule_output=y_test[i]
                coverage = str(exp.coverage())
                precision = str(exp.precision())
                
            with open(filepath,'a') as output_file:
                   output_file.write(str(test_indexes[i])+','+premise+','+coverage+','+precision+","+str(rule_output)+','+str(y_test[i])+','+str(yt[i])+'\n')
    t_end = time.time()
    print("Elapsed time [sec] - Anchors for {} test points: {}".format(len(X_test), t_end-t_start))
    
    return explanations



def ComputeMetricsForRule(verified, y_test):
    # removed .values from y_test
    tp = sum(verified & (y_test == 1))
    tn = sum(~verified & (y_test == -1))
    fn = sum(~verified & (y_test == 1))
    fp = sum(verified & (y_test == -1))
    print(f"tp = {tp}, tn = {tn}, fp = {fp}, fn = {fn}")
    precision_test = tp / (tp+fp)
    covering_test = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f1score = (2*tp)/(2*tp+fp+fn)
    error = fp/(fp+tp)
    return  precision_test, covering_test, acc, f1score, error
    
def EvaluateAnchorRule2D(X_test, y_test, op, th, feature_names):
    op1, op2 = op
    th1, th2 = th
    feature1,feature2 = feature_names
    
    verified = (operators[op1](X_test[feature1].values, th1)) & (operators[op2](X_test[feature2].values, th2))

    precision_test, covering_test, acc, f1score = ComputeMetricsForRule(verified, y_test)

    return verified,precision_test, covering_test, acc, f1score


    
def plotAnchorsOnSVM(model, X_test, y_test, border_points_idx, feature_labels,anchor_vertexes, class_labels = ["1","-1"]):
    
    anchorX1 = anchor_vertexes[0]
    anchorX2 = anchor_vertexes[1]
    
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot()
    ax.set(xlim=(0, 0.1), ylim=(0, 1))
    DecisionBoundaryDisplay.from_estimator(model,X_test.values, ax = ax, response_method="predict",plot_method="pcolormesh",cmap = "RdYlGn",alpha=0.2)
    DecisionBoundaryDisplay.from_estimator(
            model,X_test.values, ax = ax,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"]
        )
    #sc = ax.scatter(X_test["SafetyMargin"], X_test["Tau"], s = 5, c = dist, cmap = "RdYlGn")
    s1 = ax.scatter(X_test[feature_labels[0]][y_test == 1], X_test[feature_labels[1]][y_test==1], s = 5, c = "limegreen")
    s2 = ax.scatter(X_test[feature_labels[0]][y_test == -1], X_test[feature_labels[1]][y_test==-1], s = 5, c = "red")

    sc2 = ax.scatter(X_test.values[list(border_points_idx),0],X_test.values[list(border_points_idx),1], s=25, facecolors="yellow", edgecolors="k")


    '''
    # anchor 1
    r = ax.axvline(x = 0.05, ymin = 0, ymax = 0.75, c = "b")
    ax.axhline(y = 0.75, xmin = 0.5, xmax = 1, c = "b")
    # anchor 2
    ax.axvline(x = 0.03, ymin = 0, ymax = 0.52, c = "b")
    ax.axhline(y = 0.52, xmin = 0.3, xmax = 1, c = "b")

    # anchor 3: SafetyMargin > 0.05
    ax.axvline(x = 0.05, ymin = 0, ymax = 1, c = "b")
    #ax.axhline(y = 0.26, xmin = 0, xmax = 0.5, c = "b")
    '''

    r = ax.fill(anchorX1, anchorX2,facecolor='none', edgecolor='blue', linewidth=2)

    ax.set_xlabel(feature_labels[0], fontsize = 20)
    ax.set_ylabel(feature_labels[1], fontsize = 20)

    ax.legend([s1,s2,sc2,r], labels = [class_labels[0],class_labels[1], "selected points", "Joined Anchors from SVM"], loc="lower right")
    plt.show()

