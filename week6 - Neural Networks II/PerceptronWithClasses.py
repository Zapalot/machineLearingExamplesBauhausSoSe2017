# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:30:01 2017

@author: fbonowsk
"""
import math
import random
import numpy
import matplotlib.pyplot as plt

#a neuron that just gives input to other neurons:
class InputSynapse:
    output=1.0
    def __init__(self, initialOutput):
        self.output=initialOutput
    def setOutput(self, newVal):
        self.output=newVal
    def getOutput(self):
        return self.output
    #this could actually be interesting as it tells us how the inputs would have to change to 
    # come colsert to the desired final output
    def calculateWeightUpdate(self,learningRate,influenceOfOutputOnError ):
        return 0


class Neuron:
     #these are the Information sources of tthe neuron:
     #They must have an "getOutput" method. Add a "biasInput if you need one..
    inputs =[] 
    
    weights =[] #these are the memory of the neuron

    # Because weight updates must be calculated for all neurons before actually changing 
    # weights, we have to store them until we are finished with the calculations
    weightUpdates=[] 
    
    #
    lastOutput=0 # outputs are saved for later use in backpropagation
    
    #initialize neuron with a list of inputs, generate random weights if needed weights
    def __init__(self,initInputs,weights=None, addBiasInput=True):
        self.inputs=initInputs[:] # make a copy of the list, so it can be modified 
        #add bias input if needed
        if(addBiasInput):
            self.inputs.append(InputSynapse(1))
            
        #initialize weights randomly if none were given
        if(weights==None):
            self.weights= [random.uniform(-1,1) for i in self.inputs]
        else:
            self.weights=weights[:] #make copy for independence...
        self.weightUpdates =[0 for i in self.weights] # make a placeholder for the updates
    # conveniently add an input source and a 
    def addInput(self,input,weight):
        self.inputs.append(input)
        self.weights.append(weight)
        self.weightUpdates.append(0)

    # calculate the output (i.e. after getting new input or weights)
    def updateOutput(self):
        weightedInputSum=0
        # sum up inputs*weights
        for i in range(len(self.inputs)):
            #print("input:"+str(self.inputs[i].getOutput())+ "weight"+ str(self.weights[i]))
            weightedInputSum+= self.inputs[i].getOutput()*self.weights[i]
        self.lastOutput= 1.0/(1.0+math.exp(weightedInputSum))
        #print("sum:"+str(weightedInputSum)+"\toutput"+str(self.lastOutput))
        return self.lastOutput

    # return precalculated output
    def getOutput(self):
        #print("output:"+str(self.lastOutput  ))
        return self.lastOutput    


    #back-propagation: this way we tell an input that outputting something else would be better
    def calculateWeightUpdate(self,learningRate,influenceOfOutputOnError ):
        activationDerivative=self.lastOutput*(1.0-self.lastOutput)# derivative of the 1/(1+exp(x))
        activationDerivative+=0.1 #add a small number to avoid getting stranded on flat regions of the weight space
        
        # influence of this neuron on the error:
        influenceOfInputSumOnError=influenceOfOutputOnError*activationDerivative
        #all the inputs are multiplied by the same value: 
        learnMultiplier=-learningRate*influenceOfInputSumOnError;
       
        #update the weights one by one, taking the current input value into account
        for i in range( len (self.inputs)): 
            inVal = self.inputs[i].getOutput()
            self.weightUpdates[i]+=inVal*learnMultiplier
  #!!!!here come the magic part: we tell the input neurons to perform a weight update themselves
            influenceOfThisInputOnError=self.weights[i]*influenceOfInputSumOnError
            self.inputs[i].calculateWeightUpdate(learningRate,influenceOfThisInputOnError )


    #apply the precalculated weightupdates
    def applyWeightUpdate(self):
        for i in range( len (self.inputs)): 
            self.weights[i]+=self.weightUpdates[i]
            self.weightUpdates[i]=0
        
        

    
####################################################
# plug together a  perceptron with two layers and two units in the hidden layer:
####################################################
inputList=[1,1] #two components

#the inputs of the network
inputSynapses=[InputSynapse(val) for val in inputList]

#first Layer of Neurons:
firstLayerWidth=1 
#create a list of neurons that use the input synapses as inputs
firstLayer=[Neuron(inputSynapses) for index in range(firstLayerWidth)] 

#second Layer of Neurons:
outputLayerWidth=1 
#create a list of neurons that use the input synapses as inputs
outputLayer=[Neuron(firstLayer) for index in range(outputLayerWidth)] 

####################################################
# calculate the output of the network
####################################################
allLayers=[firstLayer, outputLayer]

def applyNetworkOnInputData(inputs,neuronLayers, data):
    # prepare the data to be fed into the network
    for i in range(len(inputs)):
        inputs[i].setOutput(data[i])
    # update network calculations
    layerIndex=0
    for layer in neuronLayers:
        #print("=========Layer"+str(layerIndex))
        layerIndex+=1
        for neuron in layer:
            neuron.updateOutput()
    output=[neuron.getOutput() for neuron in neuronLayers[-1]]
    return output

####################################################
#visualization
####################################################
def visualizeOutput(inputs,neuronLayers, minX,maxX,minY,maxY):
    mapsize=20 # resolution of the output imge
    outputs=numpy.zeros((mapsize,mapsize)) #initialize storage for output image
    for x in range(mapsize):
        testX=(x/mapsize)*(maxX-minX)+minX # map pixel index to coordinate between min and max
        for y in range (mapsize):
            testY=(y/mapsize)*(maxY-minY)+minY
            
            outputs[y,x]=applyNetworkOnInputData(inputs,neuronLayers, [testX,testY])[0] # save outut of neuron at that position
    # plot the output.
    # extent=(minX,maxX,minY,maxY) gives the rectagle of the inputspace covered by the data
    # origin='lower' mirrors the Y-axis so that output[0][0]appears in the lower left corner
    plt.imshow(outputs,extent=(minX,maxX,minY,maxY),origin='lower')

# show state of randomly initialized model
visualizeOutput(inputSynapses,allLayers, -20,20,-20,20)

####################################################
#training
####################################################
#this time, we show the network all training data points before we draw graphics...


def trainNetwork(learnRate,inputs,neuronLayers, trainData, targetOutputs):
    #show each datapoint once:
    for dataIndex in range(targetOutputs.shape[0]):
        curData=trainData[dataIndex,:]
        curOutputs=applyNetworkOnInputData(inputs,neuronLayers,curData)
        
        #tell the output layer what we wanted from it
        # the back propagation will pass on that information to the layers before it
        outputLayer=neuronLayers[-1]
        for outIndex in range(len(outputLayer)):
            targetOutputDiff=curOutputs[outIndex]-targetOutputs[dataIndex,outIndex]
            outputLayer[outIndex].calculateWeightUpdate(learnRate,targetOutputDiff)
        # now that the updates have been calculated, we can apply them
        layerIndex=0
        for layer in neuronLayers:
            print("===Layer"+str(layerIndex))
            layerIndex+=1
            neuronIndex=0
            for neuron in layer:
                neuron.applyWeightUpdate()
                print("=Neuron"+str(neuronIndex))
                neuronIndex+=1
                print(neuron.weights)


##run a learing loop:
nEpochs=1000 #how many times to show all datapoints
learnRate=0.1 # Learning Rate multiplier

from sklearn import datasets
iris = datasets.load_iris()         # get built in dataset to play with
trainingData=iris.data[:,[1,2]]     # use first two features for training
targetOutputs=numpy.array([float(target==0) for target in iris.target]).reshape((trainingData.shape[0],outputLayerWidth)) #all the flowers that we want to pick are marked by a '1' all ohers by a '0'


for i in range(nEpochs):
    trainNetwork(learnRate,inputSynapses,allLayers,trainingData,targetOutputs)
    plt.cla()
    visualizeOutput(inputSynapses,allLayers, -10,10,-10,10)
    plt.scatter(trainingData[:,0],trainingData[:,1],c=targetOutputs, cmap='binary')
    plt.pause(0.01) # this gives the plot an opportunity to draw itself