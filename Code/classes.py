from imports import *
from functions import *

class NeuralNetwork():
    def __init__(self, input_size, output_size, activation_funcs, activation_ders, cost_fun, cost_der, l1=True):

        self.input_size = input_size
        self.output_size = output_size
        self.act_func = activation_funcs
        self.act_der = activation_ders
        if l1:
            self.cost_fun = self.cost_l1
            self.cost_der = self.costl1_der
        else:
            self.cost_der = cost_der
            self.cost_fun = cost_fun
        self.training_info = {"Cost_history": []}
        
        self.weights = self.create_layers_batched()

    def create_layers_batched(self):
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
    
    def feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.weights, self.act_func):
            layer_inputs.append(a)
            z = np.matmul(a, W) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def backpropagation_batched(self, inputs, targets):
        layer_inputs, zs, predict = self.feed_forward_saver(inputs)
        layer_grads = [() for _ in self.weights]

        delta = self.cost_der(predict, targets) * self.act_der[-1](zs[-1])
        
        for i in reversed(range(len(self.weights))):
            layer_input = layer_inputs[i]
            (W, b) = self.weights[i]
            
            dC_dW = np.matmul(layer_input.T, delta)
            dC_db = np.sum(delta, axis=0, keepdims=True)
            
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
    
    def update_weights(self, layers_grad, learning_rate):
        k = 0
        for (W, b), (W_g, b_g) in zip(self.weights, layers_grad):
            W -= learning_rate * W_g
            b -= learning_rate * b_g
            self.weights[k] = (W, b)
            k += 1

    # Adam replacement for update_weights() - (In progress)
    def NN_Adam(self, layers_grad, learning_rate, epsilon, beta1=0.9, beta2=0.999):
        k = 0

    # momentum replacement for udate_weights() - Needs testing!
    def NN_mom(self, layers_grad, learning_rate, gamma):
        k = 0
        change_W = 0
        change_b = 0
        for (W, b), (W_g, b_g) in zip(self.weights, layers_grad):
            new_change_W = learning_rate*W_g + gamma * change_W
            new_change_b = learning_rate*b_g + gamma * change_b
            W -= new_change_W
            b -= new_change_b
            self.weights[k] = (W, b)
            change_W = new_change_W
            change_b = new_change_b
            k += 1

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        return self.cost_fun(predictions, targets)

    def train_network(self, X_train, y_train, learning_rate=0.001, epochs=100):
        for i in range(epochs):
            layers_grad = self.backpropagation_batched(X_train, y_train)
            self.update_weights(layers_grad, learning_rate)

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
                
                gradients = self.backpropagation_batched(X_batch, y_batch)
                self.update_weights(gradients, eta)
            
            cost_per_epoch.append(self.cost(input_data, target_data))
            
        self.training_info["Cost_history"] = cost_per_epoch


    def train_ADAM(self, X_train, y_train, epochs=100, batch_size=32, 
        learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        TODO: Docstring
        """
        n_samples = X_train.shape[0]
        cost_history = []
        t = 0  

        m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]
        v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                grads = self.backpropagation_batched(X_batch, y_batch)

                t += 1  

                new_weights = []
                for idx, ((W, b), (dW, db), (mW, mb), (vW, vb)) in enumerate(zip(self.weights, grads, m, v)):
                    
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
                    m[idx] = (mW, mb)
                    v[idx] = (vW, vb)


                self.weights = new_weights

            
            cost = self.cost(X_train, y_train)
            cost_history.append(cost)

            self.training_info["Cost_history"] = cost_history
            

    def cost_l1(self, y_true, y_pred, lam=0.1):
        return np.mean((y_pred - y_true)**2) + lam * np.mean(np.abs(y_pred))
    
    
    def costl1_der(self, y_true, y_pred, lam=0.1):
        B = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / B + lam * np.sign(y_pred)  



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

