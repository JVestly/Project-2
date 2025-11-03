from imports import *
from functions import *

class NeuralNetwork():
    def __init__(self, input_size, output_size, activation_funcs, activation_ders, cost_fun=None, cost_der=None, l1=False, l2=True, lam=1e-5):

        self.input_size = input_size
        self.output_size = output_size
        self.act_func = activation_funcs
        self.act_der = activation_ders
        self.lam = lam
        self.l1 = l1
        self.l2 = l2
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.training_info = {"Cost_history": []}
        
        self.weights = self._create_layers_batched()

    def _create_layers_batched(self):
        layers = []
        i_size = self.input_size
        for layer_output_size in self.output_size:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(1, layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers
    
    def predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.weights, self.act_func):
            z = np.matmul(a, W) + b
            a = activation_func(z)
        return a
    
    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.weights, self.act_func):
            layer_inputs.append(a)
            z = np.matmul(a, W) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def _backpropagation_batched(self, inputs, targets):
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.weights]

        if self.act_func[-1].__name__ == "softmax":
           delta = self.cost_der(predict, targets)
        else:
           delta = self.cost_der(predict, targets) * self.act_der[-1](zs[-1])
        
        for i in reversed(range(len(self.weights))):
            layer_input = layer_inputs[i]
            (W, b) = self.weights[i]
            
            dC_dW = np.matmul(layer_input.T, delta)
            if self.l1: 
                dC_dW += 2 * self.lam * np.sign(W)
            elif self.l2:
                dC_dW += 2 * self.lam * W
            dC_db = np.sum(delta, axis=0, keepdims=True)
            if self.l1: 
                dC_db += 2 * self.lam * np.sign(b)
            elif self.l2:
                dC_db += 2 * self.lam * b
            
            layer_grads[i] = (dC_dW, dC_db)
            
            if i > 0:
                dC_da = np.matmul(delta, W.T)
                delta = dC_da * self.act_der[i-1](zs[i-1])
                
        return layer_grads
    
    def feed_forward(self, a):
        for (W, b), activation_func in zip(self.weights, self.act_func):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def _update_weights(self, layers_grad, learning_rate):
        k = 0
        for (W, b), (W_g, b_g) in zip(self.weights, layers_grad):
            W -= learning_rate * W_g
            b -= learning_rate * b_g
            self.weights[k] = (W, b)
            k += 1

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        return self.cost_fun(predictions, targets)

    def train_network(self, X_train, y_train, learning_rate=0.001, epochs=100):
        for i in range(epochs):
            layers_grad = self._backpropagation_batched(X_train, y_train)
            self._update_weights(layers_grad, learning_rate)

    def train_SGD(self, input_data, target_data, n_epochs=1000, eta=0.1, minibatch_size=10):
        
        data_size = input_data.shape[0]
        iterations_per_epoch = int(data_size / minibatch_size)
        
        all_indices = np.arange(data_size)
        cost_per_epoch = []

        for epoch in range(n_epochs):
            
            np.random.shuffle(all_indices)
            
            for i in range(iterations_per_epoch):
                start = i * minibatch_size
                stop = (i + 1) * minibatch_size
                
                batch_indices = all_indices[start:stop]
                
                X_batch = input_data[batch_indices]
                y_batch = target_data[batch_indices]
                
                gradients = self._backpropagation_batched(X_batch, y_batch)
                self._update_weights(gradients, eta)
            
            cost_per_epoch.append(self.cost(input_data, target_data))
            
        self.training_info["Cost_history"] = cost_per_epoch


    def train_SGD(self, input_data, target_data, epochs=1000, batch_size=32, learning_rate=0.1, functional=False):
        
        n_samples = input_data.shape[0]
        cost_history = []
        if functional:
            param1 = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]
            param2 = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]
            t = 0

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = input_data[batch_idx]
                y_batch = target_data[batch_idx]
                
                gradients = self._backpropagation_batched(X_batch, y_batch)

                if functional:
                    t += 1
                    new_weights = []
                    for idx, ((W, b), (dW, db), (param1_W, param1_b), (param2_W, param2_b)) in enumerate(zip(self.weights, gradients, param1, param2)):
                        functional(t, idx, new_weights, learning_rate, W, b, dW, db, param1_W, param1_b, param2_W, param2_b, param1, param2)
                    self.weights = new_weights

                else:
                    self._update_weights(gradients, learning_rate)
            
            cost_history.append(self.cost(input_data, target_data))
        self.training_info["Cost_history"] = cost_history
            

    def cost_l1(self, y_true, y_pred):
        return mse(y_true, y_pred) + self.lam * np.sum(np.sum(np.abs(W)) for W,_ in self.weights)
    
    
    def costl1_der(self, y_true, y_pred):
        return mse_der(y_true, y_pred)
    
    
    def cost_l2(self, y_true, y_pred):
        mse_term = mse(y_true, y_pred)
        reg_term = self.lam * sum(np.sum(W**2) for W, _ in self.weights)
        return mse_term + reg_term

    
    def costl2_der(self, y_true, y_pred):
        return mse_der(y_true, y_pred)
    
    
    def check_autograd(self):
        return None


class GradientDescent:
    """
    Gradient descent class.
    Methods performind different types of Gradient descent. 
    """
    def __init__(self, X_norm, iters, y_centered, eps=1e-6, l1=False):
        self._X = X_norm
        self._iters = iters
        self._eps = eps
        self._y = y_centered
        self._m = X_norm.shape[1]
        self._l1 = l1


    
    def gradOrd(self, iters=None, eta=0.1, l=0):
        """
        Ordinary gradient descent. Can be used for both 
        OLS and Ridge, since lam=0 by default.
        Also including an invariant for using LASSO. 

        Parameters
        ----------
        eta : float, default 0.1
            Learning rate.
        l : float, default 0.0
            Ridge and LASSO penalty strength.
            
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._X.shape[1])


        for t in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=l)

            z = theta - eta * grad
            
            if self._l1:
                alpha = eta*l
                z = theta - eta*grad
                theta = soft_threshold(z, alpha)
                
            theta = z
            
            if t!= 0  and  self.stopping(grad):
                break
        
        return theta

    
    def pgdUpdate(self, beta, grad, eta, lam=0.1):
        """
        REDUNDANT FUNCTION?
        Proximal gradient descent used for LASSO regression.
        
        Parameters
        ----------
        lam: float, default 0.1.
            Lasso penalty.

        Returns
        -------
        ndarray of shape (m, )
            The optimized parameter vector
        """
        z = beta - eta*grad
        alpha = eta*lam

        return soft_threshold(z, alpha)
    
    
    def gradMomentum(self, momentum=0.9, eta=0.001, lam=0.0):
        """
        Gradient Descent with momentum for OLS/Ridge.
        Parameters
        ----------
        momentum : float, default 0.9
            Exponential weight on the previous update.
        eta : float, default 0.1
            Learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        thetas = []

        msError = []
        velocity = 0
        for _ in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam)
            velocity = eta*grad + momentum*velocity
            z = theta - velocity
            if self._l1:
                alpha = eta*lam
                theta = soft_threshold(z, alpha)
            else: 
                theta = z
            thetas.append(theta)
            pred = self._X@theta
            msError.append(mse(self._y, pred))
            if self.stopping(grad):
                break
  
        return theta

    
    def gradAda(self, eta=0.1, lam=0.0, eps=1e-8, theta0=None):
        """
        AdaGrad with r_t accumulation and H_t^{-1/2} scaling as in the slides.
        Parameters
        ----------
        eta : float, default 0.1
            Base learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer inside the square root.
        theta0 : array-like or None, default None
            Optional warm-start parameters.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        if theta0 is None:
            theta = np.zeros(self._m)
        else:
            theta = np.asarray(theta0).reshape(-1).copy()
        r = np.zeros_like(theta)
        for _ in range(self._iters):
            g = gradient(self._X, self._y, theta, lam=lam)
            if self.stopping(g):
                break
            r = r + g * g
            e = np.sqrt(r + eps)
            z = theta - eta * g / e
            if self._l1:
                alpha = e*lam
                theta = soft_threshold(z, alpha)
            else:
                theta = z
        
        return theta


    
    def gradADAM(self, eta=0.1, beta1=0.9, beta2=0.999, lam=0.0, eps=1e-8):
        """
        Adam optimizer for OLS/Ridge.
        Parameters
        ----------
        eta : float, default 0.001
            Base learning rate.
        beta1 : float, default 0.9
            Exponential decay for the first moment.
        beta2 : float, default 0.999
            Exponential decay for the second moment.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        t = 0
        first_moment = 0.0
        second_moment = 0.0

        for iter in range(1, self._iters+1):
            grad = gradient(self._X, self._y, theta, lam=lam)

            if self.stopping(grad):
                break
            
            else:
                first_moment = beta1*first_moment + (1-beta1)*grad
                second_moment = beta2*second_moment+(1-beta2)*grad*grad
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                update = eta*first_term/(np.sqrt(second_term)+eps)
                theta = update
    
        return theta



    def grad_stochastic_ADAM(self, learn_rate, init_guess, max_iter, n_epochs, tol, batch_size, epsilon, beta1=0.9, beta2=0.999):
        self.theta_stoch = init_guess
        X = self._X
        y = self._y

        M = batch_size   #size of each mini-batch
        m = int(X.shape[0]/M) #number of minibatches
        m_t = np.zeros(self.p)
        v_t = np.zeros(self.p)
        grad = 1
        while np.linalg.norm(grad) > tol and self.t < max_iter:
            for epoch in range(1, n_epochs+1):
                for i in range(m):
                    random_index = np.random.randint(X.shape[0]-M)
                    X_, y_ = X[random_index: random_index+M, :], y[random_index: random_index+M]
                    g_t = X_.T @ (X_ @ self.theta_stoch - y_) / self.n
                    m_t = beta1*m_t + (1-beta1)*g_t
                    v_t = beta2*v_t + (1-beta2)*(g_t**2)
                    m_hat = m_t / (1-beta1**self.t+epsilon)
                    v_hat = v_t / (1-beta2**self.t+epsilon)
                    # Update parameters theta
                    self.theta_stoch -= learn_rate*m_hat / (np.sqrt(v_hat)+epsilon)
                    self.t += 1
            grad = X.T @ (X @ self.theta_stoch - y) / self.n


        
    def gradStoc(self, batch_size=1, eta=0.1, lam=0.0, shuffle=True):
        """
        Stochastic (mini-batch) Gradient Descent for OLS/Ridge.
        Parameters
        ----------
        batch_size : int, default 1
            Number of samples per update step.
        eta : float, default 0.1
            Learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        shuffle : bool, default True
            Whether to shuffle indices between passes.
        Returns
        -------
        ndarray of shape (n_features,)
            The optimized parameter vector.
        """
        n_samples = self._X.shape[0]
        n_features = self._X.shape[1]
        minibatch_size = max(1, int(batch_size))

        theta = np.zeros(n_features)      
        data_indices = np.arange(n_samples)    
        steps = 0                   

        x0 = 5
        x1 = 10
        eta = x0/x1    

        while steps < self._iters:
            if shuffle:
                np.random.shuffle(data_indices)

            for start_idx in range(0, n_samples, minibatch_size):
                if steps >= self._iters:
                    break

                    
                batch_idx = data_indices[start_idx:start_idx + minibatch_size]
                X_batch = self._X[batch_idx]
                y_batch = self._y[batch_idx]

                grad = gradient(X_batch, y_batch, theta, lam=lam)
                if self.stopping(grad):
                    break

                z = theta - eta * grad
                if self._l1:
                    alpha = eta*lam
                    z = theta - eta*grad
                    theta = soft_threshold(z, alpha)
                else:
                    theta = z

                steps += 1
                #eta = self.scale_eta(steps, x0,x1)#. Using dynamic step size. 

        return theta

    
    def gradRMS(self, eta=0.01, rho=0.9, lam=0.0, eps=1e-8):
        """
        RMSprop for OLS/Ridge.
        Parameters
        ----------
        eta : float, default 0.01
            Base learning rate.
        rho : float, default 0.9
            Exponential decay for the squared-gradient accumulator.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        s = np.zeros_like(theta)
        for _ in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=lam)
            if self.stopping(grad):
                break
            s = rho * s + (1.0 - rho) * (grad * grad)
            z =  (eta / (np.sqrt(s) + eps)) * grad
            theta -= z
            
            if self._l1:
                alpha = (eta / (np.sqrt(s) + eps)) * lam
                theta -= soft_threshold(z, alpha)


        return theta

    
    def stopping(self, grad, e=None):
        """
        Early-stopping criterion based on the Euclidean norm of the gradient.
        Parameters
        ----------
        grad : array-like
            Current gradient vector.
        e : float or None, default None
            Absolute tolerance. Uses the instance tolerance when None.
        Returns
        -------
        bool
            True if the gradient norm is below tolerance, else False.
        """
       
        return float(np.linalg.norm(grad)) < self._eps
    

if __name__ == "__main__":
    from sklearn import datasets
    from classes import NeuralNetwork
    from functions import *
    from autograd import grad

    np.random.seed(50)
    n = 1000
    x = np.linspace(-1, 1, n)
    y = runge(x) + np.random.normal(0, 0.01, n)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    p = 10
    X = polynomial_features(x_train, p)
    Y = polynomial_features(y_train, p)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print(x_train.shape)
    print(x_train.reshape(-1,1).shape)
    X_train = scaler.fit_transform(x_train.reshape(-1,1))
    inputs, y_scaled = scale(X, y)
    targets = (scale(Y, y))[0]
    y_mean = np.mean(y)
    y_centered = y_train - y_mean

    m = len(targets)
    k = len(inputs)

    NNLinReg = NeuralNetwork(input=X_train, activation_ders=[sigmoid_der, sigmoid_der, identity_der], activation_funcs=[sigmoid, sigmoid, identity], input_size=k, output_size=[k,k])
    NNLinReg2 = NeuralNetwork(input=X_train, activation_ders=[sigmoid_der, sigmoid_der], activation_funcs=[sigmoid, sigmoid], input_size=k, output_size=[k,k])


    weights = NNLinReg2.train_network(y_train, epochs=10)
    predictions = NNLinReg2.feed_forward()

    # --- 2D Runge Plot ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_surface(X, Y, predictions, cmap='plasma', rstride=5, cstride=5) 

    # ax.set_title("Predictions vs 2D Runge")
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis (f(x, y))')

    plt.plot(x_train,predictions.squeeze())
    plt.scatter(x,y)
    plt.show()

