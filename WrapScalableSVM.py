import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

import qpsolvers
from qpsolvers import solve_qp
import cvxopt
from cvxopt import matrix, solvers

import joblib
from Utils_SSVM import *
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split

# custom (scalable) classifier with sklearn syntax

class ScalableSVMClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, eta, kernel, param, tau, solver = 'osqp'):

        # these are the actual parameters required
        self.eta = eta
        self.kernel = kernel
        self.param = param
        self.tau = tau
        
        if solver not in qpsolvers.available_solvers:
            print(f"{solver} is not a supported solver")
        else:
            self.solver = solver
        
    def fit(self, X, y):

        #self.Xtr = X
        #self.Ytr = y
        # if method is not classic, we need to keep n_c calibration samples apart from the training process
        '''
        if self.method!="classic":
            n_c = int(np.ceil((7.47) / self.epsilon * np.log(1 / self.delta)))
            self.Xtr,self.Xcal,self.Ytr, self.Ycal = train_test_split(X,y,test_size = n_c,random_state = 12)
        else:
            self.Xtr = X
            self.Ytr = y
        '''
        self.Xtr = X
        self.Ytr = y
        
        self.classes_ = np.unique(y)
        #print(self.Xtr.shape)
        #print(self.Ytr.shape)
        self.alpha = SSVM_Train(self.Xtr, self.Ytr, self.kernel, self.param, self.tau, self.eta, solver = self.solver)
        self.b = offset(self.Xtr, self.Ytr, self.alpha, self.kernel, self.param, self.eta, self.tau)
        
        return self

    def FPcontrol(self, Xcal, Ycal, epsilon, method):

        """Given a method, computes the right scaling parameter b_eps """
        
         # if method and epsilon are None, classic SVM
        #self.b_eps = 0
        #self.epsilon = epsilon
        #self.delta = delta
        if method not in ["classic","cp","ps"]:
            print(f"{method} is not a supported method")
        else:
            self.method = method

        if self.method == "ps":

            self.b_eps = b_epsPS(self.Xtr,self.Ytr, Xcal, Ycal, self.b, self.alpha, self.kernel, self.param, self.eta, epsilon)

        elif self.method == "cp":

            self.b_eps = b_epsCP(self.Xtr,self.Ytr, Xcal, Ycal, self.b, self.alpha, self.kernel, self.param, self.eta, epsilon)

        elif self.method == "classic":

            self.b_eps = 0

        return self
    
    def get_params(self, deep=True):
    
        return {"eta": self.eta,"kernel":self.kernel, "param": self.param,"tau": self.tau,"solver":self.solver}#,"method":self.method}#, "epsilon":self.epsilon, "delta":self.delta}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict(self, X):
        return -np.sign(self.decision_function(X)).reshape((len(X),))
    
    def decision_function(self, X):
        """X: test data """
        '''
        if self.method!="classic":
            self.b_eps = FPcontrol(self.Xtr,self.Ytr, self.Xcal, self.Ycal,self.method, self.b, self.alpha, self.kernel, self.param, self.eta, self.epsilon)
        else:
            self.b_eps = 0
        '''

        K = KernelMatrix(self.Xtr, X, self.kernel, self.param)

        return -self.b -self.eta*(np.matmul(K.T,np.matmul(np.diag(self.Ytr.squeeze()),self.alpha))) + self.b_eps 