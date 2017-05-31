import matplotlib.pyplot as plt


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy
#import some flowers from sklearn
from sklearn import datasets
iris = datasets.load_iris()         # get built in dataset to play with
trainingData=iris.data[:,[1,2]]     # use first two features for training
targetOutputs=numpy.array([float(target==0) for target in iris.target]) #all the flowers that we want to pick are marked by a '1' all ohers by a '0'

#We build a neural network layer by layer in "keras sequential mode"
model = Sequential()
# In Sequential mode, layers are added one by one
#
# "Dense"
model.add(Dense(1, activation='sigmoid', input_dim=1))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # create an optimizer that will fit weights
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

# lets show the map from last time:
def visualizeOutput(model,minX,maxX,minY,maxY):
    # yet another way to build the data grids necessary for visualization:
    stepsize=0.1 #spacing between datapoints
    xSteps=numpy.arange( minX,maxX, stepsize) # make a list of values between min and max
    ySteps=numpy.arange( minY,maxY, stepsize)
    xMesh, yMesh=numpy.meshgrid(xSteps,ySteps) # build a "image like" array of x- and y values
    mapPositionsAsTable=numpy.c_[xMesh.ravel(), yMesh.ravel()] # flatten those images into a long column and glue x and y together
#!
    neuralPredictions=model.predict(mapPositionsAsTable) # get the output of the neural network
    # outputs have one row per datapoint
    predictionMap=neuralPredictions.reshape(xMesh.shape) # reshape the output vector into an image 
    
    # plot the output.
    # extent=(minX,maxX,minY,maxY) gives the rectagle of the inputspace covered by the data
    # origin='lower' mirrors the Y-axis so that output[0][0]appears in the lower left corner
    plt.imshow(predictionMap,extent=(minX,maxX,minY,maxY),origin='lower')


#visualize the training process:
while True:
    # train with one round of all datapoints

#!
    model.fit(trainingData, targetOutputs,epochs=1,batch_size=2)
    plt.cla() # clean up plot
    # show decicion boundary
    visualizeOutput(model,trainingData[:,0].min(), trainingData[:,0].max(),trainingData[:,1].min(), trainingData[:,1].max())
    # show data used for training
    plt.scatter(trainingData[:,0],trainingData[:,1],c=targetOutputs,cmap='binary')
    # wait a bit to let the graphics refresh
    plt.pause(0.05)