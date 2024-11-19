import numpy as np
import pandas as pd
# PREPROCESSING AND DATA EXTRACTION UTILS 

def mix_gauss(mu, Sigma, n):
    d = mu.shape[1]
    p = mu.shape[0]

    X = np.zeros((0, d))
    Y = np.zeros((0,))

    for i in range(p):
        Xi = np.zeros((n, d))
        Yi = np.zeros((n,))
        for j in range(n):
            x = np.random.multivariate_normal(mu[i], np.diag(Sigma), 1)
            Xi[j, :] = x
            Yi[j] = i
        X = np.vstack((X, Xi))
        Y = np.hstack((Y, Yi))

    return X, Y

def split_dataset(X, Y, train_size, test_size, calib_size, save_path):

    train_size = int(train_size)
    test_size = int(test_size)
    calib_size = int(calib_size)

   
    if train_size <= 0 or test_size <= 0 or calib_size <= 0:
        raise ValueError('Sizes of training, test, and calibration sets must be positive integers.')

    total_samples = X.shape[0]
    total_set_size = train_size + test_size + calib_size

    if total_set_size > total_samples:
        raise ValueError('Sizes of training, test, and calibration sets exceed the total number of samples.')

    
    perm_indices = np.random.permutation(total_samples)

    X_train = X[perm_indices[:train_size]]
    Y_train = Y[perm_indices[:train_size]]

    X_test = X[perm_indices[train_size:train_size + test_size]]
    Y_test = Y[perm_indices[train_size:train_size + test_size]]

    X_calib = X[perm_indices[train_size + test_size:train_size + test_size + calib_size]]
    Y_calib = Y[perm_indices[train_size + test_size:train_size + test_size + calib_size]]

    train_df = pd.DataFrame(X_train, columns=["SafetyMargin","Eta","Tau","meanEntropy","medianEntropy","stdsEntropy","iqrsEntropy"])
    train_df['output'] = Y_train
    
    test_df = pd.DataFrame(X_test, columns=["SafetyMargin","Eta","Tau","meanEntropy","medianEntropy","stdsEntropy","iqrsEntropy"])
    test_df['output'] = Y_test
    
    calib_df = pd.DataFrame(X_calib, columns=["SafetyMargin","Eta","Tau","meanEntropy","medianEntropy","stdsEntropy","iqrsEntropy"])
    calib_df['output'] = Y_calib

    train_df.to_csv(f'{save_path}/train.csv', index=False)
    test_df.to_csv(f'{save_path}/test.csv', index=False)
    calib_df.to_csv(f'{save_path}/calib.csv', index=False)
    return print("Split Done")
    
    train_df = pd.DataFrame(X_train, columns=["x","y"])
    train_df['output'] = Y_train
    
    test_df = pd.DataFrame(X_test, columns=["x","y"])
    test_df['output'] = Y_test
    
    calib_df = pd.DataFrame(X_calib, columns=["x","y"])
    calib_df['output'] = Y_calib

    train_df.to_csv(f'{save_path}/train.csv', index=False)
    test_df.to_csv(f'{save_path}/test.csv', index=False)
    calib_df.to_csv(f'{save_path}/calib.csv', index=False)
    return print("Split Done")
    
# def split_dataset(X, Y, train_size, test_size, calib_size):
    
#     train_size = int(train_size)
#     test_size = int(test_size)
#     calib_size = int(calib_size)

#     if train_size <= 0 or test_size <= 0 or calib_size <= 0:
#         raise ValueError('Sizes of training, test, and calibration sets must be positive integers.')

#     total_samples = X.shape[0]
#     total_set_size = train_size + test_size + calib_size

#     if total_set_size > total_samples:
#         raise ValueError('Sizes of training, test, and calibration sets exceed the total number of samples.')

#     perm_indices = np.random.permutation(total_samples)

#     X_train = X[perm_indices[:train_size]]
#     Y_train = Y[perm_indices[:train_size]]

#     X_test = X[perm_indices[train_size:train_size + test_size]]
#     Y_test = Y[perm_indices[train_size:train_size + test_size]]

#     X_calib = X[perm_indices[train_size + test_size:train_size + test_size + calib_size]]
#     Y_calib = Y[perm_indices[train_size + test_size:train_size + test_size + calib_size]]

#     return X_train, Y_train, X_test, Y_test, X_calib, Y_calib

def generate_data(N_points, p_A, p_O, mu1, mu2, Sigma1, Sigma2):
    p_B = 1 - p_A

    X_red = np.zeros((0, mu1.shape[1]))
    Y_red = np.zeros((0,))
    X_blue = np.zeros((0, mu2.shape[1]))
    Y_blue = np.zeros((0,))

    for i in range(N_points):
        if np.random.rand() < p_A:
            if np.random.rand() < p_O:  # outlier
                x_i, y_i = mix_gauss(mu1, Sigma1, 1)
                y_i[y_i == 0] = -1
            else:
                x_i, y_i = mix_gauss(mu1, Sigma1, 1)
                y_i[y_i == 0] = 1

            X_red = np.vstack((X_red, x_i))
            Y_red = np.hstack((Y_red, y_i))
        else:
            if np.random.rand() < p_O:  # outlier
                x_i, y_i = mix_gauss(mu2, Sigma2, 1)
                y_i[y_i == 0] = 1
            else:
                x_i, y_i = mix_gauss(mu2, Sigma2, 1)
                y_i[y_i == 0] = -1

            X_blue = np.vstack((X_blue, x_i))
            Y_blue = np.hstack((Y_blue, y_i))

    X = np.vstack((X_red, X_blue))
    Y = np.hstack((Y_red, Y_blue))

    return X, Y