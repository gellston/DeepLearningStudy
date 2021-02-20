# one neuron without activation

# for one sample (x, y)
# s = wx + b
# e = (s-y)^2
# de/dw = ds/dw de/ds = x 2(s-y)
# de/db = ds/db de/ds = 1 2(s-y)

# for batch samples
# (grad_w, grad_b) = average (de/dw, de/db) over samples in batch
# because loss function for the batch is the average of the loss of each sample

# update
# w -= grad_w * learning_rate
# b -= grad_b * learning_rate

import numpy as np
import matplotlib.pyplot as plt


class FNN:

	def __init__(self, lr=0.01):
		self.lr = lr
		# initial weights
		self.w = 0.5; self.b = 0.

	def predict(self, x): # forward propagation
		return self.w * x + self.b

	# train for one batch
	def train_on_batch(self, x, y):

		# batch forward propagation
		s = self.w * x + self.b

		# These will be summed over the batch
		grad_sum_w = 0.; grad_sum_b = 0.
		batch_size = x.shape[0]

		for n in range(batch_size):
			# de/db = 1 2(s-y)
			# de/dw = x 2(s-y)
			dedb = 2 * (s[n]-y[n])
			dedw = x[n] * dedb
			grad_sum_b += dedb; grad_sum_w += dedw

		# gradient descent
		self.w -= grad_sum_w / batch_size * self.lr
		self.b -= grad_sum_b / batch_size * self.lr

	def fit(self, x, y, batch_size, epochs, validation_data):

		errors = [] # validation loss after each epoch

		for epoch in range(epochs):

			for i in range(0, x.shape[0], batch_size):
				self.train_on_batch(x[i:i+batch_size], y[i:i+batch_size])

			# calculate mse validation loss
			y_pred = self.predict(validation_data[0])
			e = np.mean(np.square(y_pred-validation_data[1]))
			errors.append(e)
			print('w b e =', self.w, self.b, e)

		return errors


x_train = np.random.rand(1024)
# x_train = np.random.rand(1024,1)
y_train = x_train * 2 + 1

x_val = np.random.rand(32)
# x_val = np.random.rand(32,1)
y_val = x_val * 2 + 1

x_test = np.array([[0.2], [0.4], [0.6], [0.8]])
y_test = x_test * 2 + 1

fnn = FNN(0.01)
errors = fnn.fit(x_train, y_train, 32, 300, (x_val,y_val))

y_pred = fnn.predict(x_test)
print('y_pred'); print(y_pred)
print('y_test'); print(y_test)

plt.plot(errors[30:])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()