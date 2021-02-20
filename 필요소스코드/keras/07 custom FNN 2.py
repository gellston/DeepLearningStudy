# 3 neurons in series with activation f[1], f[2], f[3]

# for one sample (x, y)
# l = layer index
# s[l] = w[l] o[l-1] + b[l]
# o[l] = f[l](s[l])
# e = (o[l_last]-y)^2

# de/dw[l] = ds[l]/dw[l] do[l]/ds[l] de/do[l]
#          = o[l-1] f'[l](s[l]) de/do[l]
#            If l is not the last layer
#          = o[l-1] f'[l](s[l]) do[l+1]/do[l] de/do[l+1]
#          = o[l-1] f'[l](s[l]) ds[l+1]/do[l] do[l+1]/ds[l+1] de/do[l+1]
#          = o[l-1] f'[l](s[l]) w[l+1] f'[l+1](s[l+1]) de/do[l+1]

# delta[l] = f'[l](s[l]) de/do[l]
#            If l is not the last layer
#          = f'[l](s[l]) w[l+1] delta[l+1]

# de/dw[l] = o[l-1] delta[l]
# de/db[l] = delta[l], if we set o[l-1] to 1 for bias

# de/do[l_last] = 2 (o[l_last]-y)

# for batch samples
# (grad_w, grad_b) = average (de/dw, de/db) over samples in batch
# because loss function for the batch is the average of the loss of each sample

# update
# w[l] -= grad_w[l] * learning_rate
# b[l] -= grad_b[l] * learning_rate

import numpy as np
import matplotlib.pyplot as plt


class FNN:

	def __init__(self, lr=0.01):
		self.lr = lr # learning rate
		self.w = []; self.b = [] # weights
		self.f = []; self.f_deriv = [] # activations and their derivatives
		self.N_layers = 0

	def add(self, activation=None, activation_deriv=None):
		# initial weights
		self.w.append(np.random.rand()*2-1)
		self.b.append(0.)
		# activation function and its derivative
		self.f.append(activation)
		self.f_deriv.append(activation_deriv)
		self.N_layers += 1

	def propagate_forward(self, x):
		s = [x]; o = [x]
		for l in range(1, self.N_layers):
			s.append( self.w[l] * o[l-1] + self.b[l] )
			o.append( self.f[l](s[l]) )
		return s, o

	def predict(self, x):
		s, o = self.propagate_forward(x)
		return o[-1] # output from the last layer

	# train for one batch
	def train_on_batch(self, x_batch, y_batch):

		Nl = self.N_layers
		l_last = Nl - 1

		# batch forward propagation
		s_batch, o_batch = self.propagate_forward(x_batch)

		# These will be summed over the batch
		grad_sum_w = [0.] * Nl
		grad_sum_b = [0.] * Nl
		batch_size = x_batch.shape[0]

		for n in range(batch_size):

			# get n-th sample
			s = []; o = []; y = y_batch[n]
			for l in range(Nl):
				s.append( s_batch[l][n] )
				o.append( o_batch[l][n] )

			# delta[l_last] = f'[l_last](s[l_last]) 2(o[l_last]-y)
			# delta[l] = f'[l](s[l]) w[l+1] delta[l+1]
			delta = [0.] * Nl
			delta[l_last] = self.f_deriv[l_last](s[l_last]) * 2 * (o[l_last]-y)
			for l in range(l_last-1, 0, -1):
				delta[l] = self.f_deriv[l](s[l]) * self.w[l+1] * delta[l+1]

			# de/dw[l] = o[l-1] delta[l]
			# de/db[l] = delta[l]
			for l in range(1, Nl):
				dedw = o[l-1] * delta[l]
				dedb = delta[l]
				grad_sum_w[l] += dedw
				grad_sum_b[l] += dedb

		# gradient descent
		for l in range(1, Nl):
			self.w[l] -= grad_sum_w[l] / batch_size * self.lr
			self.b[l] -= grad_sum_b[l] / batch_size * self.lr

	def fit(self, x, y, batch_size, epochs, validation_data):
		errors = [] # validation loss after each epoch

		for epoch in range(epochs):
			for i in range(0, x.shape[0], batch_size):
				self.train_on_batch(x[i:i+batch_size], y[i:i+batch_size])
			# calculate mse validation loss
			y_pred = self.predict(validation_data[0])
			e = np.mean(np.square(y_pred-validation_data[1]))
			errors.append(e)
			print('w =', self.w[1:])
			print('b =', self.b[1:])
			print('e =', e)

		return errors


def linear(x):  # linear y = x
	return x


def linear_deriv(x):  # derivative of y = x
	return 1


def sigmoid(x):  # sigmoid
	return 1. / (1 + np.exp(-x))


def sigmoid_deriv(x):  # derivative of sigmoid
	return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):  # hyperbolic tangent
	return np.tanh(x)


def tanh_deriv(x):  # derivative of hyperbolic tangent
	return 1 - np.tanh(x) ** 2


x_train = np.random.rand(1024)
#x_train = np.random.rand(1024,1)
y_train = x_train * 0.1 - 0.05

x_val = np.random.rand(32)
#x_val = np.random.rand(32,1)
y_val = x_val * 0.1 - 0.05

x_test = np.array([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]])
y_test = x_test * 0.1 - 0.05

fnn = FNN(0.1)
fnn.add() # input layer
fnn.add(linear, linear_deriv) # hidden layer
fnn.add(sigmoid, sigmoid_deriv) # hidden layer
fnn.add(tanh, tanh_deriv) # output layer
fnn.w = [0.0, 0.0, 0.5, 1.0]

errors = fnn.fit(x_train, y_train, 32, 1000, (x_val,y_val))

y_pred = fnn.predict(x_test)
print('y_pred'); print(y_pred)
print('y_test'); print(y_test)

plt.plot(errors[30:])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()