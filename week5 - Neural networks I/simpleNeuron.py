# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:45:13 2017

@author: fbonowsk
"""
import numpy
import math
import matplotlib.pyplot as plt
import random
# many of the functions here could be much shorter if you use numpy arrays and dot-products
# instead of looping over lists!

#######################
# This function calculates the output of a single neuron for a single input
def calculateOutput(inputs, weights):
    weightedInputSum=0
    # sum up inputs*weights
    for i in range(len(inputs)):
        weightedInputSum+= inputs[i]*weights[i]
    return (1.0/(1+math.exp(2*weightedInputSum))) # sigmoid activation function

#######################
# This function calculates the output of a single neuron with a range of inputs
# as the inputs are varied along two dimensions (x/y), it creates a map of responses
# that is visualized using matplotlib
def visualizeOutput(weights, minX,maxX,minY,maxY):
    mapsize=50 # resolution of the output imge
    outputs=numpy.zeros((mapsize,mapsize)) #initialize storage for output image
    for x in range(mapsize):
        testX=(x/mapsize)*(maxX-minX)+minX # map pixel index to coordinate between min and max
        for y in range (mapsize):
            testY=(y/mapsize)*(maxY-minY)+minY
            outputs[y,x]=calculateOutput([testX,testY,1.0],weights) # save outut of neuron at that position
    # plot the output.
    # extent=(minX,maxX,minY,maxY) gives the rectagle of the inputspace covered by the data
    # origin='lower' mirrors the Y-axis so that output[0][0]appears in the lower left corner
    plt.imshow(outputs,extent=(minX,maxX,minY,maxY),origin='lower')

###############################################################
# play around with the weights a bit to see what they do.
# note how the decision boundary prescribes a circle around the origin
# the the distance to the origin is given by the "bias" weight
#
# The loop will go on forever -you can stop the program
# by pressing "ctrl-C" or the dark red square "stop buton" next to the console
################################################################

# prepare the plot
plotRegion=[0,6,0,7] # minx, maxX, minY,maxY
plt.axis(plotRegion)
#plt.ion();
t=0# we use this as a timer
weights=[1.0,1.0,1.0] # weights for x, y and bias/threshold
while False: #continue forever, can be stopped with the dark red square "stop buton" next to the console
#for i in range(5):
    #update weights to rotate the decision boundary
    weights[0]=math.sin(t)
    weights[1]=math.cos(t)
    t+=0.05
    
    plt.cla(); # clean old data from plot to improve performance
    visualizeOutput(weights, plotRegion[0],plotRegion[1],plotRegion[2],plotRegion[3])
    plt.pause(0.01) # this gives the plot an opportunity to draw itself

##################################################
#Let' try to learn the difference between two flowers from the 'iris' dataset
# we use only the first two features to make visualization easier
from sklearn import datasets
iris = datasets.load_iris()         # get built in dataset to play with
trainingData=iris.data[:,[1,2]]     # use first two features for training
targetOutputs=[float(target==0) for target in iris.target] #all the flowers that we want to pick are marked by a '1' all ohers by a '0'
plotRegion=[trainingData[:,0].min(),trainingData[:,0].max(),trainingData[:,1].min(),trainingData[:,1].max()] # minx, maxX, minY,maxY

##################################################
#a function for calculating a weight update by steepest descent
# targetOutputDifference: the difference between desired and actual output
# output: the ouput of the neuron
# input: a list of inputs of this neuron

#note: this could be written much shorter, but I preferred to make it easier to read
def getWeightUpdate(learningRate, targetOutputDifference, currentOutput, currentInputs):
    activationDerivative=currentOutput*(1-currentOutput)# derivative of the 1/(1+exp(x))
    activationDerivative+=0.1 #add a small number to avoid getting stranded on flat regions of the weight space
    
    #all the inputs are multiplied by the same value: 
    multiplier=-learningRate*targetOutputDifference*activationDerivative;
    weightUpdates=[]
    for inVal in currentInputs:
        weightUpdates.append(inVal*multiplier)
    return weightUpdates

def trainNeuron(inputs, weights, targetOutput, learningRate):
    output=calculateOutput(inputs, weights)
    weightUpdate=getWeightUpdate(learningRate,targetOutput-output, output, inputs)
    for i in range(len(weights)):
        weights[i]+=weightUpdate[i]


# now we choose random samples and train the network with them
while True:
    sampleIndex=random.randrange(0,len(targetOutputs))
    sampleInput= numpy.append(trainingData[sampleIndex,:],1.0)
    sampleTargetOutput=targetOutputs[sampleIndex]
    trainNeuron(sampleInput, weights, sampleTargetOutput,0.02)
    
    plt.cla(); # clean old data from plot to improve performance
    visualizeOutput(weights, plotRegion[0],plotRegion[1],plotRegion[2],plotRegion[3])
    plt.scatter(trainingData[:,0],trainingData[:,1],c=targetOutputs, cmap='binary')
    plt.pause(0.01) # this gives the plot an opportunity to draw itself
    
    # see whoich one it got right or wrong:
    curOuts=[]
    errSum=0
    for i in range(len(targetOutputs)):
        curOut=calculateOutput(numpy.append(trainingData[i,:],1.0),weights)
        errSum+=(curOut-targetOutputs[i])*(curOut-targetOutputs[i])
        curOuts.append(calculateOutput(numpy.append(trainingData[i,:],1.0),weights))
    print(errSum)

