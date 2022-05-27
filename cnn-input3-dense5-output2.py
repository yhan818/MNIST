import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential([
        Dense(5, input_shape=(3,),  activation='relu'),
        Dense(2, activation='softmax'),
])

a = np.array([0,1,1])
print(a.shape)

# matplotlib here

img= np.expand_dims(ndimage.imread('NN.PNG',0))
plt.imshow(img[0])


