from imports import *
from numpy.linalg import pinv

def mse(y_true, y_pred):
    """TODO: Complete docstring. Reuse code from P1"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. y: {y_true.shape}, p:{y_pred.shape}")

    if y_true.ndim == 1:
        n = len(y_true)
        error_sum = 0.0
        for i in range(n):
            error_sum += (y_true[i] - y_pred[i])**2

        return error_sum / n


    elif y_true.ndim == 2:
        n, m = y_true.shape 
        mse_total = 0.0

        for j in range(m): 
            error_sum = 0.0

            for i in range(n): 
                error_sum += (y_true[i, j] - y_pred[i, j])**2
            mse_total += error_sum / n 

        return mse_total / m  

    else:
        raise ValueError("Inputs must be 1D or 2D arrays")
    

def ols(X, y):
    """
    Ordinary Least Squares regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.

    Returns
    -------
    ndarray, shape (n_features,)
        Estimated regression coefficients (beta).
    """
    return (pinv(X.T @ X)) @ X.T @ y


def ridge(X, y, lam=0.1):
    """
    Ridge regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.
    lam : float, default=0.1
        Regularization strength (lambda).

    Returns
    -------
    ndarray, shape (n_features,)
        Estimated regression coefficients (beta).
    """
    n_features = X.shape[1]

    return np.linalg.pinv(X.T @ X + lam * np.eye(n_features)) @ X.T @ y


def soft_threshold(z, alpha):
    """
    Used element-wise in gradient descent for Lasso regression.
    Shrinks large values, and sets small values to zero.

    Parameters
    ----------
    z : ndarray
        Predicted betas.
    alpha : float
        Threshold value.

    Returns
    -------
    float 
        Estimated regression coefficient (beta).
        Returns 0 if the absoulte value of y is less than or equal to alpha
    """
        
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)
    
def polynomial_features(x, p, intercept=False):
    """
    Generate a polynomial feature matrix from input data.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Input feature values.
    p : int
        Polynomial degree.
    intercept : bool, default=False
        If True, includes a column of ones for the intercept term.

    Returns
    -------
    ndarray, shape (n_samples, p) or (n_samples, p+1)
        Design matrix with polynomial features up to degree p.
    """
    n = len(x)

    if intercept:
        X = np.zeros((n, p + 1))

        for i in range(p + 1):
            X[:, i] = x**i

        return X
    
    X = np.zeros((n, p))

    for i in range(1, p + 1):
        X[:, i - 1] = x**i

    return X


def gradient(X, y, beta, lam=0.0):
    """
    Compute the gradient of the cost function for linear regression.

    Supports both Ordinary Least Squares (OLS) and Ridge regression
    depending on the value of the regularization parameter `lam`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.
    beta : ndarray, shape (n_features,)
        Current estimate of regression coefficients.
    lam : float, default=0.0
        Regularization parameter.
        - lam = 0.0: OLS gradient
        - lam > 0.0: Ridge gradient

    Returns
    -------
    ndarray, shape (n_features,)
        Gradient of the cost function with respect to `beta`.
    """
    n = X.shape[0]
    if lam != 0.0:
        return (2 / n) * X.T @ ((X @ beta) - y) + 2 * lam * beta

    return (2 / n) * X.T @ ((X @ beta) - y)
        

import numpy as np

def softmax(z):
    """Changed to cope with ZeroDivisionError"""
    z = z - np.max(z, axis=1, keepdims=True)      
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def cross_entropy(pred, targets, eps=1e-12):
    p = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(targets * np.log(p), axis=1))


def cross_entropy_der(pred, targets):
    return (pred - targets) / targets.shape[0]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def softmax_der(z):
    """TODO: Write docstring"""
    s = softmax(z)
    N, C = s.shape
    J = -np.einsum('bi,bj->bij', s, s)   
    idx = np.arange(C)
    J[:, idx, idx] += s                  
    return J



def sigmoid(z):
    """TODO: docstring"""
    return 1 / (1 + np.exp(-z))


def sigmoid_der(z):
    sig_diff = np.e**(-z)/(1+np.e**(-z))**2
    return sig_diff


def reLU(z):
    """TODO: Doc. String"""
    return np.where(z > 0, z, 0)


def leaky_reLU(z, alpha=0.0001):
    """TODO: docstring"""
    return np.where(z > np.zeros(z.shape), z, alpha*z)


def leaky_reLU_der(z, alpha=0.0001):
    "TODO: ---"
    return np.where(z > 0, 1, alpha)


def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def identity(x):
    """Identity function used for Linear Regression.
    """
    return x


def identity_der(x):
    return 1


def mse_der(predict, target):
    r = 2/len(predict)*(predict - target)
    return r


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


def runge(x):
    return 1 / (25*x**2 + 1)


def runge2d(x,y):
    return 1 / ((10*x-5)**2 + (10*y-5)**2 + 1)


def scale(X, y):
    """Homemade scaling function.
    
    Parameters
    ----------
    X: array-like
        Polynomial feature matrix (design matrix)
    
    y: array
        List of y_values (generic)

    Returns 
    -------
    Scaled X and y based on z-score normalization
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_centered = y-y_mean

    return X_norm, y_centered


def NN_Moment(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2, gamma=0.3):

    change_W = param1_W
    change_b = param1_b
    new_change_W = learning_rate*dW + gamma*change_W
    new_change_b = learning_rate*db + gamma*change_b
    W -= new_change_W
    b -= new_change_b
    new_weights.append((W, b))
    param1[idx] = (change_W, change_b)  


def NN_RMS(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2, beta=0.9, epsilon=1e-8):

    vW = param1_W
    vb = param1_b
    vW = beta * vW + (1 - beta) * (dW ** 2)
    vb = beta * vb + (1 - beta) * (db ** 2)

    W -= learning_rate * dW / (np.sqrt(vW) + epsilon)
    b -= learning_rate * db / (np.sqrt(vb) + epsilon)

    new_weights.append((W, b))
    param1[idx] = (vW, vb)


def NN_ADAM(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2, beta1=0.9, beta2=0.999, epsilon=1e-8):

    mW = param1_W
    mb = param1_b
    vW = param2_W
    vb = param2_b

    mW = beta1 * mW + (1 - beta1) * dW
    mb = beta1 * mb + (1 - beta1) * db
    vW = beta2 * vW + (1 - beta2) * (dW ** 2)
    vb = beta2 * vb + (1 - beta2) * (db ** 2)

    mW_hat = mW / (1 - beta1 ** t)
    mb_hat = mb / (1 - beta1 ** t)
    vW_hat = vW / (1 - beta2 ** t)
    vb_hat = vb / (1 - beta2 ** t)

    W -= learning_rate * mW_hat / (np.sqrt(vW_hat) + epsilon)
    b -= learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

    new_weights.append((W, b))
    param1[idx] = (mW, mb)
    param2[idx] = (vW, vb)


def create_and_scale_data(state=50, n=1000, noise_std=0.01):
    """TODO: Docstring"""
    np.random.seed(state)

    n = 1000
    x = np.linspace(-1, 1, n).reshape(-1, 1)
    y = 1/(1 + 25 * x**2) +  np.random.normal(0, noise_std, x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=state)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(x_train.reshape(-1,1))
    Y_train = scaler_y.fit_transform(y_train.reshape(-1,1))

    X_test = scaler_X.transform(x_test.reshape(-1,1))
    Y_test = scaler_y.transform(y_test.reshape(-1,1))

    return X_train, X_test, Y_train, Y_test, x_train, x_test, y_train, y_test

def create_and_scale_dataP1(state=50, n=1000, noise_std=0.01):

    np.random.seed(state)
    x = np.linspace(-1, 1, n)
    y = runge(x) +  np.random.normal(0, noise_std, n)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=state)

    n = 10
    X = polynomial_features(x_train2, n)
    Y = polynomial_features(x_test2, n)


    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X)
    Xte_s = scaler.transform(Y)

    return Xtr_s, Xte_s, x_train2, x_test2, y_train2, y_test2