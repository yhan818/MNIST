# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization

# load train and test dataset
def load_dataset():
	# load dataset
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# reshape dataset to have a single channel
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
	# one hot encode target values
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	return X_train, y_train, X_test, y_test

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()

	# 32 filters of 3x3 convolution on 28x28 b/w image. 3x3 filter initially random. model.shape == (None, 28,28, 32)
	# Parameter 3 x 3 x 1 depth * 32 + 1 bias * 32 = 320 ;
	layer1=Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))
	model.add(layer1)
	# adding one convolution layer will increase accurate rate 0.5% but with 2x of time running
	# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

	# batch normalization changing the distribution of the output of the layer. accelerating the learning
	# Some reports suggest better performance. Seems no mathematical proof. Empirical only.
	# model.add(BatchNormalization())

	model.add(MaxPooling2D((2, 2)))              # pooling layer
	model.add(Flatten())

	# No convolution layer. result can be 97%
	#model.add(Flatten(input_shape=(28,28,1)))

	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))   # full-connected (Dense) output layer, transfer to the possibilities
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	model.summary()

	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		X_train, y_train, X_test, y_test = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

		# trains the model for a fixed number of epochs.
		history = model.fit(X_train, y_train,  batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test) )

		# evaluate model
		_, acc = model.evaluate(X_test, y_test, verbose=1)
		print('> %.2f' % (acc * 100.0))

		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.2f std=%.2f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# run the test harness for evaluating a model
def run_test():
	# load dataset
	X_train, y_train, X_test, y_test = load_dataset()
	# prepare pixel data
	X_train, X_test = prep_pixels(X_train, X_test)
	# evaluate model
	scores, histories = evaluate_model(X_train, y_train)

	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test()