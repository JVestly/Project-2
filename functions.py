import numpy as np
from sklearn.metrics import accuracy_score

def mse():
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
        

def cross_entropy(target, predict):
    """TODO: Complete docstring"""
    return np.sum(-target * np.log(predict))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def sigmoid(z):
    """TODO: docstring"""
    return 1 / (1 + np.exp(-z)) 


def reLU(target, predict):
    """TODO: Doc. String"""
    return np.sum(-target * np.log(predict))


def leaky_reLU(z, alpha=0.1):
    """TODO: docstring"""
    return alpha*(np.exp(z)-1) if z < 0 else z


def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def mse_der(predict, target):
    r = (2/len(predict))*(predict - target)
    return r


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


def runge2d(x,y):
    return 1 / ((10*x-5)**2 + (10*y-5)**2 + 1)