from imports import *
from numpy.linalg import pinv

def mse(y_true, y_pred):
    """
    Mean squared error (MSE) between predictions and true values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
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
    """
    Compute the softmax of a 2D input array.

    Parameters
    ----------
    z : array-like
        Input matrix.

    Returns
    -------
    array-like
        Softmax probabilities for each row.
    """
    z = z - np.max(z, axis=1, keepdims=True)      
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def cross_entropy(pred, targets, eps=1e-12):
    """
    Cross-entropy loss.

    Parameters
    ----------
    pred : array-like
        Predicted probabilities.
    targets : array-like
        One-hot encoded target labels.
    eps : float, default 1e-12
        Small value to avoid log(0).

    Returns
    -------
    float
        Cross-entropy loss.
    """
    p = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(targets * np.log(p), axis=1))


def cross_entropy_der(pred, targets):
    """
    Derivative of cross-entropy loss.

    Parameters
    ----------
    pred : array-like
        Predicted probabilities.
    targets : array-like
        One-hot encoded target labels.

    Returns
    -------
    array-like
        Gradient of the loss.
    """
    return (pred - targets) / targets.shape[0]


def softmax_vec(z):
    """
    Compute softmax for a 1D vector.

    Parameters
    ----------
    z : array-like
        Input vector.

    Returns
    -------
    array-like
        Softmax probabilities.
    """
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def softmax_der(z):
    """
    Compute the derivative (Jacobian) of the softmax function.

    Parameters
    ----------
    z : array-like
        Input matrix.

    Returns
    -------
    array-like
        Jacobian matrix for each sample.
    """
    s = softmax(z)
    N, C = s.shape
    J = -np.einsum('bi,bj->bij', s, s)   
    idx = np.arange(C)
    J[:, idx, idx] += s                  
    return J



def sigmoid(z):
    """
    Sigmoid activation function.

    Parameters
    ----------
    z : array-like
        Input values.

    Returns
    -------
    array-like
        Output of sigmoid.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_der(z):
    """
    Derivative of the sigmoid function.

    Parameters
    ----------
    z : array-like
        Input values.

    Returns
    -------
    array-like
        Derivative of sigmoid.
    """
    sig_diff = np.exp(-z)/(1+np.e**(-z))**2
    return sig_diff


def reLU(z):
    """
    ReLU activation function.

    Parameters
    ----------
    z : array-like
        Input values.

    Returns
    -------
    array-like
        Output of ReLU.
    """
    return np.where(z > 0, z, 0)


def leaky_reLU(z, alpha=0.1):
    """
    Leaky ReLU activation function.

    Parameters
    ----------
    z : array-like
        Input values.
    alpha : float, default 0.1
        Slope for negative inputs.

    Returns
    -------
    array-like
        Output of Leaky ReLU.
    """
    return np.where(z > np.zeros(z.shape), z, alpha*z)


def leaky_reLU_der(z, alpha=0.1):
    """
    Derivative of Leaky ReLU.

    Parameters
    ----------
    z : array-like
        Input values.
    alpha : float, default 0.1
        Slope for negative inputs.

    Returns
    -------
    array-like
        Derivative values.
    """
    return np.where(z > 0, 1, alpha)


def ReLU_der(z):
    """
    Derivative of the ReLU activation.

    Parameters
    ----------
    z : array-like
        Input values.

    Returns
    -------
    array-like
        Derivative values.
    """
    return np.where(z > 0, 1, 0)


def identity(x):
    """
    Identity activation function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Output equal to input.
    """
    return x


def identity_der(x):
    """
    Derivative of the identity function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    int
        Always 1.
    """
    return 1


def tanh(x):
    """
    Hyperbolic tangent activation function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Output of tanh.
    """
    return np.tanh(x)


def tanh_der(x):
    """
    Derivative of the tanh activation.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Derivative of tanh.
    """
    return 1.0 - np.tanh(x)**2


def ELU(x, alpha=0.01):
    """
    Exponential Linear Unit activation.

    Parameters
    ----------
    x : array-like
        Input values.
    alpha : float, default 0.01
        Scale for negative region.

    Returns
    -------
    array-like
        Output of ELU.
    """
    return np.where(x < 0, alpha*(np.exp(x)-1), x)


def ELU_der(x, alpha=0.01):
    """
    Derivative of ELU activation.

    Parameters
    ----------
    x : array-like
        Input values.
    alpha : float, default 0.01
        Scale for negative region.

    Returns
    -------
    array-like
        Derivative of ELU.
    """
    return np.where(x<0, ELU(x) + alpha, 1)


def GELU(x):
    """
    Gaussian Error Linear Unit activation.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Output of GELU.
    """
    return 0.5*x *(tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))) 


def GELU_der(x):
    """
    Derivative of the GELU activation.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Derivative of GELU.
    """
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2))) + (x * np.exp(-x**2 / 2)) / np.sqrt(2 * np.pi)


def mse_der(predict, target):
    """
    Derivative of mean squared error.

    Parameters
    ----------
    predict : array-like
        Predicted values.
    target : array-like
        True values.

    Returns
    -------
    array-like
        Gradient of MSE.
    """
    r = 2/len(predict)*(predict - target)
    return r


def accuracy(predictions, targets):
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : array-like
        Model outputs.
    targets : array-like
        True labels (one-hot encoded).

    Returns
    -------
    float
        Accuracy score.
    """
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


def runge(x):
    """
    The Runge function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        1 / (25x^2 + 1)
    """
    return 1 / (25*x**2 + 1)


def runge2d(x,y):
    """
    2D Runge function.

    Parameters
    ----------
    x : array-like
        x-values.
    y : array-like
        y-values.

    Returns
    -------
    array-like
        1 / ((10x-5)^2 + (10y-5)^2 + 1)
    """
    return 1 / ((10*x-5)**2 + (10*y-5)**2 + 1)


def onehot(y, n=None):
    """
    Convert integer labels to one-hot encoding.

    Parameters
    ----------
    y : array-like
        Integer labels.
    n : int or None, default None
        Number of classes.

    Returns
    -------
    array-like
        One-hot encoded labels.
    """
    y = np.asarray(y, dtype=int).ravel()
    n = np.max(y) + 1 if n is None else n
    m = np.zeros((y.size, n))
    m[np.arange(y.size), y] = 1
    return m


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
    """
    Momentum update for neural network weights.

    Parameters
    ----------
    t, idx : int
        Iteration counters.
    new_weights : list
        Updated weights storage.
    learning_rate : float
        Learning rate.
    W, b : array-like
        Current weights and biases.
    dW, db : array-like
        Current gradients.
    param1_W, param1_b : array-like
        Previous momentum terms.
    param2_W, param2_b : ignored
        Placeholder for RMS/Adam compatibility.
    param1, param2 : list
        Parameter storage.
    gamma : float, default 0.3
        Momentum factor.

    Returns
    -------
    None
    """
    change_W = param1_W
    change_b = param1_b
    new_change_W = learning_rate*dW + gamma*change_W
    new_change_b = learning_rate*db + gamma*change_b
    W -= new_change_W
    b -= new_change_b
    new_weights.append((W, b))
    param1[idx] = (change_W, change_b)  


def NN_RMS(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2, beta=0.9, epsilon=1e-8):
    """
    RMSprop weight update.

    Parameters
    ----------
    Same as NN_Moment, with:
    beta : float, default 0.9
        Decay rate.
    epsilon : float, default 1e-8
        Stability constant.

    Returns
    -------
    None
    """
    vW = param1_W
    vb = param1_b
    vW = beta * vW + (1 - beta) * (dW ** 2)
    vb = beta * vb + (1 - beta) * (db ** 2)

    W -= learning_rate * dW / (np.sqrt(vW) + epsilon)
    b -= learning_rate * db / (np.sqrt(vb) + epsilon)

    new_weights.append((W, b))
    param1[idx] = (vW, vb)


def NN_ADAM(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimizer update rule.

    Parameters
    ----------
    Same as NN_RMS, with:
    beta1 : float, default 0.9
        First moment decay.
    beta2 : float, default 0.999
        Second moment decay.
    epsilon : float, default 1e-8
        Stability constant.

    Returns
    -------
    None
    """
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
    """
    Create 1D Runge data and standardize it.

    Parameters
    ----------
    state : int, default 50
        Random seed.
    n : int, default 1000
        Number of samples.
    noise_std : float, default 0.01
        Standard deviation of Gaussian noise added to targets.

    Returns
    -------
    tuple
        (X_train, X_test, Y_train, Y_test, x_train, x_test, y_train, y_test)
        Scaled training and test sets, and their unscaled counterparts.
    """
    np.random.seed(state)

    x = np.linspace(-1, 1, n).reshape(-1, 1)
    y = runge(x) +  np.random.normal(0, noise_std, x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=state)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(x_train.reshape(-1,1))
    Y_train = scaler_y.fit_transform(y_train.reshape(-1,1))

    X_test = scaler_X.transform(x_test.reshape(-1,1))
    Y_test = scaler_y.transform(y_test.reshape(-1,1))

    return X_train, X_test, Y_train, Y_test, x_train, x_test, y_train, y_test


def create_and_scale_dataP1(state=50, n=1000, noise_std=0.01):
    """
    Create and scale data for polynomial regression using the Runge function.

    Parameters
    ----------
    state : int, default 50
        Random seed.
    n : int, default 1000
        Number of samples.
    noise_std : float, default 0.01
        Standard deviation of Gaussian noise added to targets.

    Returns
    -------
    tuple
        (Xtr_s, Xte_s, x_train, x_test, y_train, y_test)
        Scaled polynomial feature matrices for train/test, and raw data.
    """
    np.random.seed(state)
    x = np.linspace(-1, 1, n)
    y = runge(x) +  np.random.normal(0, noise_std, n)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=state)

    p = 10
    X = polynomial_features(x_train2, p)
    Y = polynomial_features(x_test2, p)


    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X)
    Xte_s = scaler.transform(Y)

    return Xtr_s, Xte_s, x_train2, x_test2, y_train2, y_test2

