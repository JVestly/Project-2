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
            z = np.matmul(a, W.T) + b
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
            z = np.matmul(a, W.T) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    
    def backpropagation_batched(self, input, layers, target, cost_der=mse_der):
        layer_inputs, zs, predict = self.feed_forward_saver_batched(input, layers)
        layer_grads = [() for layer in layers]

        for i in reversed(range(len(layers))):
            layer_input, z, self.act_der = layer_inputs[i], zs[i], self.act_der[i]
        
            if i == len(layers) - 1:
                dC_da = cost_der(predict, target)
            else:
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T
            dC_dz = dC_da * self.act_der(z)
            dC_da = dC_dz @ W.T
            dC_dW = np.matmul(layer_input.T,dC_dz)

            dC_db = dC_dz
            layer_grads[i] = (dC_dW, dC_db)
        return layer_grads
    
                              
    def cost(self, input, target):
        predict = self.feed_forward_saver(self.input, self.act_der)
        return mse(predict, target)
    
    
    def update_weights(self,  layers_grad, learning_rate):
        for (W, b), (W_g, b_g) in zip(self.weights, layers_grad):
            W -= learning_rate*W_g
            b -= learning_rate*b_g

        return W,b
    
    
    def train_network(self, targets, learning_rate=0.001, epochs=100):

        for i in range(epochs):
            layers_grad = self.backpropagation_batched(self.input, self.weights, targets)

        return self.update_weights(self.weights, layers_grad, learning_rate)
    


class GradientDescent():
    """Reuse/redo sgd on RMS and ADAM"""
    def __init__(self, dummy):
        self.d = dummy


if __name__ == "__main__":
    from sklearn import datasets
    from classes import NeuralNetwork
    from functions import accuracy, sigmoid, softmax_vec
    from autograd import grad

    iris = datasets.load_iris()
    inputs = iris.data
    targets = targets = np.zeros((len(iris.data), 3))
    for i, t in enumerate(iris.target): 
        targets[i, t] = 1

    NNLinReg = NeuralNetwork(sigmoid_der, sigmoid)

    network_input_size = 4
    np.random.seed(50)
    activation_funcs = [sigmoid, softmax_vec]
    layer_output_sizes = [4,3]
    layers = NNLinReg.create_layers_batched(network_input_size, layer_output_sizes)
    print(layers[:5])
    NNLinReg.train_network(inputs, layers, activation_funcs, targets)
    print(layers[:5])
    predictions = NNLinReg.feed_forward_saver_batched(inputs, layers, activation_funcs)
    print(predictions.shape)
    print(accuracy(predictions, targets))