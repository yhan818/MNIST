# example of loading the mnist dataset


from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

def load_dataset():
# load dataset
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# summarize loaded dataset
	print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
	print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
	print(X_train[1])

	# plot first few images
	plt.imshow(X_train[7777], cmap='gray')

	for i in range(20):
		# define subplot
		plt.subplot(5, 4,  1 + i)
		# plot raw pixel data
		plt.imshow(X_train[i], cmap='gray')

	plt.show()



load_dataset()
