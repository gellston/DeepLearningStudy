# 2->3->3->2 with activation f[l]j

# for one sample (x, y)
# l = layer index
# N[l] = number of neurons in layer l
# N[0] = 2, N[1] = 3, N[2] = 3, N[3] = 2
# index [l]j = jth neuron in layer l
# s[l]j = sum_i=1 to N[l-1] { w[l]ij o[l-1]i } + b[l]j
# o[l]j = f[l]j(s[l]j)
# e = sum_j=1 to N[l_last] { (o[l_last]j - yj)^2 } / N[l_last]

# de/dw[l]ij = ds[l]j/dw[l]ij do[l]j/ds[l]j de/do[l]j
#            = o[l-1]i f'[l]j(s[l]j) de/do[l]j
#              If l is not the last layer
#            = o[l-1]i f'[l]j(s[l]j) sum_k=1 to N[l+1] { do[l+1]k/do[l]j de/do[l+1]k }
#            = o[l-1]i f'[l]j(s[l]j) sum_k=1 to N[l+1] { ds[l+1]k/do[l]j do[l+1]k/ds[l+1]k de/do[l+1]k }
#            = o[l-1]i f'[l]j(s[l]j) sum_k=1 to N[l+1] { w[l+1]jk f'[l+1]k(s[l+1]k) de/do[l+1]k }

# delta[l]j = f'[l]j(s[l]j) de/do[l]j
#             If l is not the last layer
#           = f'[l]j(s[l]j) sum_k=1 to N[l+1] { w[l+1]jk delta[l+1]k }
#           = f'[l]j(s[l]j) matmul(w[l+1][j], delta[l+1])
#           = f'[l]j(s[l]j) matmul(w[l+1], delta[l+1])[j]

# de/dw[l]ij = o[l-1]i delta[l]j
# de/db[l]j = delta[l]j, if we set o[l-1]i to 1 for bias

# de/do[l_last]j = 2 ( o[l_last]j - yj ) / N[l_last]

# for batch samples
# (grad_w, grad_b) = average (de/dw, de/db) over samples in batch
# because loss function for the batch is the average of the loss of each sample

# update
# w[l]ij -= grad_w[l]ij * learning_rate
# b[l]j  -= grad_b[l]j  * learning_rate

import numpy as np
import matplotlib.pyplot as plt


class FNN:

	def __init__(self, lr=0.01):
		self.lr = lr # learning rate
		self.w = []; self.b = []  # weights
		self.f = []; self.f_deriv = []  # activations and their derivatives
		self.N_layers = 0

	def add(self, units, activation=None, activation_deriv=None):
		# initial weights
		input_dim = self.w[-1].shape[1] if self.N_layers > 0 else 1
		self.w.append(np.random.rand(input_dim, units) * 2 - 1)
		self.b.append(np.zeros(units))
		# activation function and its derivative
		self.f.append(activation)
		self.f_deriv.append(activation_deriv)
		self.N_layers += 1

	def propagate_forward(self, x):
		s = [x]; o = [x]
		for l in range(1, self.N_layers):
			s.append( np.matmul(o[l-1], self.w[l]) + self.b[l] )
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

			# delta[l_last]j = f'[l_last]j(s[l_last]j) 2(o[l_last]j - yj)/N[l_last]
			# delta[l]j = f'[l]j(s[l]j) matmul(w[l+1], delta[l+1])[j]
			delta = [np.zeros(1)] * Nl
			delta[l_last] = self.f_deriv[l_last](s[l_last]) * 2 * (o[l_last]-y) / y.shape[-1]
			for l in range(l_last-1, 0, -1):
				delta[l] = self.f_deriv[l](s[l]) * np.matmul(self.w[l+1], delta[l+1])

			# de/dw[l]ij = o[l-1]i delta[l]j
			# de/db[l]j = delta[l]j
			for l in range(1, Nl):
				dedw = np.zeros(self.w[l].shape, np.float32)
				for i in range(dedw.shape[0]):
					for j in range(dedw.shape[1]):
						dedw[i,j] = o[l-1][i] * delta[l][j]
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
			print(epoch, 'e =', e)

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


x_train = np.random.rand(1024,2)
y_train = np.array([ [ x[0]+x[1], x[0]-x[1] ] for x in x_train ]) * 0.1

x_val = np.random.rand(32,2)
y_val = np.array([ [ x[0]+x[1], x[0]-x[1] ] for x in x_val ]) * 0.1

x_test = np.array([[0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1]])
y_test = np.array([ [ x[0]+x[1], x[0]-x[1] ] for x in x_test ]) * 0.1

fnn = FNN(1.)
fnn.add(2) # input layer
fnn.add(3, linear, linear_deriv) # hidden layer
fnn.add(3, sigmoid, sigmoid_deriv) # hidden layer
fnn.add(2, tanh, tanh_deriv) # output layer

errors = fnn.fit(x_train, y_train, 20, 1000, (x_val,y_val))

y_pred = fnn.predict(x_test)
print('y_pred'); print(y_pred)
print('y_test'); print(y_test)

plt.plot(errors[25:])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()