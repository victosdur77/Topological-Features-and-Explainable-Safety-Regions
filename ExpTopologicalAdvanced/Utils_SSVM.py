import numpy as np
import scipy
from scipy import linalg
import qpsolvers
from qpsolvers import solve_qp
import scipy.sparse as sp

# SSVM UTILS 

def SquareDist(X1, X2):
    n = X1.shape[0]
    m = X2.shape[0]

    sq1 = np.sum(X1 * X1, axis=1)
    sq2 = np.sum(X2 * X2, axis=1)
    
    D = np.outer(sq1, np.ones(m)) + np.outer(np.ones(n), sq2) - 2 * np.dot(X1, X2.T)
    
    return D

def KernelMatrix(X1, X2, kernel, param):

    if kernel == 'linear':
        K = np.matmul(X1, X2.T)
    elif kernel == 'polynomial':
        K = (1 + np.matmul(X1, X2.T)) ** param
    elif kernel == 'gaussian':
        K = np.exp(-1 / (2 * param**2) * SquareDist(X1, X2))
    else:
        raise ValueError("Invalid kernel type")
    
    return K

# SSVM TRAINING 

def is_positive_definite(matrix):
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False

    # Calculate eigenvalues
    eigenvalues, _ = np.linalg.eig(matrix)

    # Check if all eigenvalues are greater than zero
    if np.all(eigenvalues > 0):
        return True
    else:
        return False

def SSVM_Train(Xtr, Ytr, kernel, param, tau, eta, solver = "osqp"):
    
    n = Xtr.shape[0]
    
    Ytr = Ytr.squeeze() # Ytr must be 1-dimensional (n,)
    
    tau = np.sort(tau)
    m = tau.shape[1]

    K = KernelMatrix(Xtr, Xtr, kernel, param)
    D = np.diag(Ytr)

    H1 = D @ K @ D #@: same of np.matmul(a,b)
    O = np.zeros((n, n * m))
    OO = np.zeros((n * m, n * m))

    H = 1 * eta * np.block([[H1, O], [O.T, OO]])
    f = np.concatenate((np.ones(n), np.zeros(n * m)))

    lb = np.zeros(n * (m + 1))
    ub = np.ones(n * (m + 1))
    ub[:n] = np.inf

    for k in range(m):
        ub[n * (k+1):n * (k + 2)] = 0.5 * ((1 - 2 * tau[:, k]) * Ytr + 1)
        
    A1 = np.eye(n)
    A2Cell = [-np.ones((1, m))] * n  # Creating a list of length n, each element being an array of shape (1, m)
    A2 = scipy.linalg.block_diag(*A2Cell)  # Creating a block diagonal matrix using the list of arrays

    Aeq1 = np.hstack((A1, A2))
    beq1 = np.zeros(n)

    A3 = np.zeros((m, n))
    
    A4Cell = [Ytr] * m
    A4 = scipy.linalg.block_diag(*A4Cell)

    Aeq2 = np.hstack((A3, A4))
    beq2 = np.zeros(m)
    
    Aeq = np.vstack((Aeq1, Aeq2))
    beq = np.hstack((beq1, beq2))
    
    if not is_positive_definite(H):
    
        H = 0.5*(H+H.T)

    H = sp.csc_matrix(H)
    
    Aeq = sp.csc_matrix(Aeq)
        
    sol = solve_qp(H, -f, None, None, Aeq, beq, lb, ub, solver = solver, eps_abs=1e-6,max_iter=10000)
        
    if sol is not None and sol.any():
        alpha_bar = sol[:n]
    else:
        print("No solution found!")
        return None
    
    return alpha_bar

def offset(Xtr, Ytr, alpha, kernel, param, eta, tau):
    
    Ytr = Ytr.squeeze() # Ytr must be 1-dimensional (n,)
    
    if  tau.shape[1] > 1:
        print('Cannot compute the offset, there must be only 1 tau!')
        return
    else:
        C = 0.5 * ((1 - 2 * tau) * Ytr + 1).reshape(1,Xtr.shape[0])
        thr = 0.001

        ind = np.where((alpha - (0 + thr) > 0) & (alpha - (C - thr) < 0))[1]
       
        X_SV = Xtr[ind]
        Y_SV = Ytr[ind]

        if kernel == 'linear':
            w = -eta * (alpha.T @ np.diag(Ytr)) @ Xtr     
            bii = X_SV@w.T + Y_SV.reshape(-1,1)

        else:
            K = KernelMatrix(Xtr, X_SV, kernel, param)
            D = np.diag(Ytr)
            bii = -eta * (K.T @ (D @ alpha)) + Y_SV
            
        b = np.random.choice(bii.flatten(), 1)
            
        return b
    
# FPR CONTROL 

def barrhoSVM(Xtr, Ytr, X, b, alpha, kernel, param, eta):
    
    Ytr = Ytr.squeeze()
    
    if kernel == 'linear':
        
        w = -eta * (np.matmul(alpha.T @ np.diag(Ytr), Xtr))
        r = b - np.matmul(X,w.T)
        
    else:
        
        K = KernelMatrix(Xtr, X, kernel, param)
        D = np.diag(Ytr)
        r = b + eta * np.matmul(K.T, np.matmul(D, alpha))
    
    return r

                                                # PROBABILISTIC SCALING #
    
def b_epsPS(Xtr, Ytr, Xcl, Ycl, b, alpha, kernel, param, eta, epsilon):
    
    Ycl = Ycl.squeeze()

    n_cl = Xcl.shape[0]
    r = int(np.ceil(epsilon * n_cl * 0.5))

    Xcl_U = Xcl[Ycl == -1, :]

    Gamma_rho = barrhoSVM(Xtr, Ytr, Xcl_U, b, alpha, kernel, param, eta)
    Gamma_rho_sorted = np.sort(Gamma_rho)[::-1]
    b_eps = Gamma_rho_sorted[r]

    return b_eps

                                                # CONFORMAL PREDICTION #

def scoreSVM(Xtr, Ytr, X, Y, b, alpha, kernel, param, eta):
    
    Y = Y.squeeze()
    
    return -Y*barrhoSVM(Xtr, Ytr, X, b, alpha, kernel, param, eta)

def b_epsCP(Xtr, Ytr, Xcl, Ycl, b, alpha, kernel, param, eta, epsilon):

    n_cl = Xcl.shape[0]
    
    Ycl = Ycl.squeeze()
    
    scores = scoreSVM(Xtr, Ytr, Xcl, Ycl, b, alpha, kernel, param, eta)

    qhat = np.quantile(scores, np.ceil((n_cl + 1) * (1-epsilon)) / n_cl)

    b_eps = np.abs(qhat)
    
    return b_eps


# FPR CONTROL WITH PROBABILISTIC SCALING AND CONFORMAL PREDICTION
'''
def FPcontrol(Xtr,Ytr, Xcal, Ycal,method, b, alpha, kernel, param, eta, epsilon):

    """Given a method, computes the right scaling parameter b_eps """
    
    if method == "ps":
            
        b_eps = b_epsPS(Xtr,Ytr, Xcal, Ycal, b, alpha, kernel, param, eta, epsilon)
            
    elif method == "cp":
            
        b_eps = b_epsCP(Xtr,Ytr, Xcal, Ycal, b, alpha, kernel, param, eta, epsilon)
    
    elif method == "classic":
        
        b_eps = 0
            
    return b_eps
'''


