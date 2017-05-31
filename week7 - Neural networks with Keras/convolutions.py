# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:32:04 2017

@author: fbonowsk
"""
from sklearn.datasets import load_sample_image


import numpy

import keras
from keras.layers import Conv2D
from keras.optimizers import SGD

# prepare some data:
    
convolutionMatrix=numpy.array([[[[-1,1],[-1,1]]]])
convolutionMatrix=convolutionMatrix.reshape((2,2,1,1))
image = load_sample_image("china.jpg")


# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
image = numpy.array(image, dtype=numpy.float64) / 255
image= image[:,:,1]#take only red channel
image=image.reshape(list(image.shape)+[1])
inputShape=list(image.shape)

#create a to model for convolution
model = keras.models.Sequential()
convolutionLayer=Conv2D(filters=1,kernel_size= (2, 2),input_shape=inputShape, activation='linear')
model.add(convolutionLayer)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # create an optimizer that will fit weights
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

theWeights=convolutionLayer.get_weights()

convolutionLayer.set_weights([convolutionMatrix,numpy.array([0])])


image=image.reshape([1]+inputShape)

output=model.predict(image)
import matplotlib.pyplot as plt
plt.imshow(output[0,:,:,0],cmap='binary')