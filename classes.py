from imports import *
from functions import *

class NeuralNetwork():
    """TODO: Implement FFNN code, generic for both Linear- and Logistic regression"""
    def __init__(self, input, activation_ders, activation_funcs, input_size, output_size):

        self.input = input
        self.act_der = activation_ders
        self.act_func = activation_funcs
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self.create_layers_batched()

    def create_layers(self):
        layers = []
        i_size = self.input_size
        for layer_output_size in self.output_size:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers
    
    
    def create_layers_batched(self):
        layers = []
        i_size = self.input_size
        for layer_output_size in self.output_size:
            W = np.random.rand(i_size, layer_output_size)
            b = np.random.rand(1,layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers
    

    def feed_forward(self):
        a = self.input
        for (W, b), activation_func in zip(self.weights, self.act_func):
            z = a@W + b
            a = activation_func(z)
        return a
    
    
    def feed_forward_batch(self):
        a = self.input
        for (W, b), activation_func in zip(self.weights, self.act_func):
            z = np.matmul(a, W) + b
            a = activation_func(z)
        return a
    

    def feed_forward_saver(self):
        layer_inputs = []
        zs = []
        a = self.input
        for (W, b), activation_func in zip(self.weights, self.act_func):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    
    def feed_forward_saver_batched(self):
        layer_inputs = []
        zs = []
        a = self.input
        for (W, b), activation_func in zip(self.weights, self.act_func):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    
    def backpropagation_batched(self, input, layers, target, cost_der=mse_der):
        layer_inputs, zs, predict = self.feed_forward_saver_batched()
        layer_grads = [() for layer in layers]

        for i in reversed(range(len(self.weights))):
            layer_input, z, act_der = layer_inputs[i], zs[i], self.act_der[i]
        
            if i == len(layers) - 1:
                dC_da = cost_der(predict, target)
            else:
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * act_der(z)
            dC_dW = np.matmul(layer_input.T,dC_dz)

            dC_db = np.mean(dC_dz, axis=0)
            layer_grads[i] = (dC_dW, dC_db)
        return layer_grads
    
                              
    def cost(self, input, target):
        predict = self.feed_forward_saver(self.input, self.act_der)
        return mse(predict, target)
    
    
    def update_weights(self, layers_grad, learning_rate):
        k = 0
        for (W, b), (W_g, b_g) in zip(self.weights, layers_grad):
            W -= learning_rate*W_g
            b -= learning_rate*b_g

            self.weights[k] =  W,b
            k += 1
    
    
    def train_network(self, targets, learning_rate=0.001, epochs=100):

        for i in range(epochs):
            layers_grad = self.backpropagation_batched(self.input, self.weights, targets)

        return self.update_weights(layers_grad, learning_rate)
    


class GradientDescent():
    """Reuse/redo sgd on RMS and ADAM"""
    def __init__(self, dummy):
        self.d = dummy


if __name__ == "__main__":
    from sklearn import datasets
    from classes import NeuralNetwork
    from functions import accuracy, sigmoid, softmax_vec
    from autograd import grad

    np.random.seed(50)
    n = 100
    x = np.linspace(-1, 1, n)
    y = runge(x) + np.random.normal(0, 0.1, n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    n = 6
    X = polynomial_features(x_train, n)
    Y = polynomial_features(y_train, n)
    inputs, y_scaled = scale(X, y)
    targets = (scale(Y, y))[0]
    y_mean = np.mean(y)
    y_centered = y_train - y_mean

    m = len(targets)

    NNLinReg = NeuralNetwork(input=inputs, activation_ders=[sigmoid_der, sigmoid_der, identity_der], activation_funcs=[sigmoid, sigmoid, identity], input_size=n, output_size=[n,4,n])
    NNLinReg2 = NeuralNetwork(input=inputs, activation_ders=[sigmoid_der, sigmoid_der, sigmoid_der], activation_funcs=[sigmoid, sigmoid, sigmoid], input_size=n, output_size=[n,4,n])


    NNLinReg2.train_network(targets)
    predictions = NNLinReg2.feed_forward_batch()

    plt.scatter(x_train, predictions[:, 0])
    #plt.plot(x_train, predictions[:,0])
    plt.show()