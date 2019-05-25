# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:14:40 2019

@author: Sean Lesch
"""

import time
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import special

class network:
    
    #Takes a number of hidden units to generate and momentum
    #value as arguments to ease experimentation
    def __init__(self, numHiddenUnits, momentum):
        
        #Save arguments
        self.numHiddenUnits = numHiddenUnits
        self.momentum = momentum
        
        #First open mnist data and transform it into a usable
        #format, namely 2 numpy arrays that can be further
        #separated if we wish to use a validation set.
        #Data is scaled once we begin the backprop algorithm
        #on an input, and the target is removed as well.
        f = open('mnist_train.csv', 'r')
        trainData = csv.reader(f)
        trainList = list(trainData)
        self.trainingDataFull = np.array(trainList)
        
        f1 = open('mnist_test.csv', 'r')
        testData = csv.reader(f1)
        testList = list(testData)
        self.testingDataFull = np.array(testList)
        
        #Number of output classes (number from 0-9)
        self.outputSize = 10
        
        #Initial starting weights are randomized from -0.05
        #to 0.5 as defined in the assignment statement. While
        #we would need to first remove the target class to get
        #an accurate weight array size, the bias is added for 
        #each of these layers once the target is removed so we 
        #the size overall remains the same. The output weights
        #must be adjusted though.
        self.weightsHidden = np.random.rand(np.size(self.trainingDataFull, 1),
                                            self.numHiddenUnits+1)*0.1-0.05
                                            
        self.weightsOutput = np.random.rand(self.numHiddenUnits+1,
                                            self.outputSize)*0.1-0.05
        
        #Learning rate for SGD
        self.learnRate = 0.1
        
        #The bias is defined in the assignment statement to be 1
        self.bias = 1
        
    #Trains the neural network for a number of epochs. Handles
    #the normalization of data and accuracy checking as well.
    def train(self, epochs, dataset, training):
        
        #Total targets and predictions for final confusion matrix
        totalTargets = []
        totalPredicts = []
        
        #Timing for personal reasons
        tStart = time.time()
        print(self.numHiddenUnits, 'hidden units\n',
              'Momentum:', self.momentum, '\nTraining =', training)

        #Add bias node to inputs now that we have saved the targets.
        #It simply takes the place of the target in the input.
        
        #Create empty vectors for the weights for updating the momentum
        #term in SGD. The first epoch has zero momentum.
        updateHidden = np.zeros((np.shape(self.weightsHidden)))
        updateOutput = np.zeros((np.shape(self.weightsOutput)))
            
        #Performs the backwards propagation phase. Takes the inputs and targets
        #and performs the forwardPhase as well.
        for n in range(np.size(dataset, 0)):
            
            #Get input row
            inputs = dataset[n,:]
                    
            #Isolate the target for the current batch
            target = inputs[0].astype('int')
            
            totalTargets.append(target)
    
            #Data is scaled to values from 0 to 1 as required by the assignment
            #and must be cast to floats so we can do so.
            inputs = inputs.astype('float16')/255
            
            #Add bias node to inputs now that we have saved the targets.
            #It simply takes the place of the target in the input.
            inputs[0] = self.bias                   
            
            #Before we can go backwards, we go forwards.
            output = self.forwardPhase(inputs)
            
            #Get the classes for later accuracy calculations
            totalPredicts.append(np.argmax(output))
            
            #Encode the target to match the output size and structure.
            targetArr = np.zeros((1,10))+0.1
            targetArr[0,target] = 0.9
            
            #If we are using the testing data, we do not update weights
            if(training):
                    
                #Compute the ouput layer weight updates from the derivative of the
                #sigmoid function.
                delOutputs = output * (1 - output) * (targetArr - output)
                
                #Compute the hidden layer weight updates from the updated output layer dotted
                #with the current weights at the hidden layer.
                delHidden = self.hOut * (1 - self.hOut) * (np.dot(delOutputs, self.weightsOutput.T))
                
                #Update the weights from the found weight updates, then save them 
                #for the next epoch and update the weights.
                updateOutput = (self.learnRate * delOutputs.T * self.hOut.reshape(1,np.size(self.hOut))).T + (self.momentum * updateOutput)
                updateHidden = (self.learnRate * delHidden.T * inputs.reshape(1,np.size(inputs))).T + (self.momentum * updateHidden)
                self.weightsOutput += updateOutput
                self.weightsHidden += updateHidden
                
        #Display the error
        accuracy = (np.array(totalTargets) == np.array(totalPredicts)).sum()/float(len(totalPredicts))*100
        print('Epoch', epochs)
        print('Accuracy:', accuracy)
            
         
        #Print analytics for the epoch
        print('Epoch compute time:', time.time()-tStart)
        print('Final Confusion matrix:')
        print(confusion_matrix(totalTargets, totalPredicts))
        #Write accuracy to file for later graphing
        with open('AccTrain'+str(training)+'QuarterSet.csv', 'a', newline ='') as f:
            w = csv.writer(f)
            w.writerow([epochs, accuracy])
        
        
        
    #Performs the forward phase. Returns outputs from output layer.
    def forwardPhase(self, inputs):
        #Dot product
        self.hOut = np.dot(inputs, self.weightsHidden)
        #Sigmoid 1/(1+exp(-z)) for all outputs
        self.hOut = special.expit(self.hOut)
        #Holds both outputs for later error computation
        
        #Take hidden layer outputs and feed into output layer
        outputs = np.dot(self.hOut, self.weightsOutput)
        outputs = special.expit(outputs)
        return outputs
    
        
            
nn = network(100, 0.9)
np.random.shuffle(nn.trainingDataFull)
splitArr = np.split(nn.trainingDataFull, 4)
for i in range(50):
    #Train on the training data 
    nn.train(i, splitArr[0], True)
    #After each epoch, we compute the accuracy
    #of the test data as well.
    nn.train(i, nn.testingDataFull, False)

            
            