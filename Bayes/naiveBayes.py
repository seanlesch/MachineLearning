# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:05:40 2019

@author: Sean Lesch

This program uses the Spambase dataset from the UCI ML repository.
https://archive.ics.uci.edu/ml/datasets/spambase
It implements a Naive Bayes classifier using a Gaussian distribution.
"""

import csv
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


#Implements a simplified gaussian normal after the use of the natural
#log on the function.
def natLog(sample, mean, std):
    #Avoiding div by 0 errors by setting a small std deviation as suggested
    #in the problem description.
    if std == 0:
        std = 0.0001
    norm = 1/(np.sqrt(2*np.pi)*std)
    exp = -(sample - mean)**2/(2*std**2)
    return np.log(norm)+exp

tStart = time.time()

#Open the datafile to get the data.
f = open('spambase.data','r')
reader = csv.reader(f)
r_list = list(reader)
fullData = np.array(r_list).astype(np.float)

#Data comes presorted by classification, separate targets
#from the dataset and split the data and targets proportionally
#between both sets. StratifiedShuffleSplit keeps this proportion
#to 40% not spam and 60% spam for the sets. Note that the spam
#class is 1, and not spam is 0.
targets = fullData[:,57]
data = fullData[:,:57]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

#"x" values are feautres, "y" values are targets. This loops uses
#the indexes output from the Stratified Shuffle Split to generate
#training and testing of equal size.
for trainIndex, testIndex in sss.split(data, targets):
    xTrain, xTest = data[trainIndex], data[testIndex]
    yTrain, yTest = targets[trainIndex], targets[trainIndex]

print("Data processing time:", time.time()-tStart, "s")

#Compute priors and verify integrity of the proportions in the 
#dataset. Should be 39.4% spam and 60.6% not spam.
priorSpam = np.count_nonzero(yTrain ==1)/len(yTrain)
priorNotSpam = np.count_nonzero(yTrain==0)/len(yTrain)
print("Class distribution:\n %.4f%%" " Spam\n %.4f%% Not Spam" % (priorSpam*100, priorNotSpam*100))

tTrain = time.time()

#For each feature within a class from the training set, we compute 
#the mean and standard deviation.
meanSpam = np.mean(xTrain[yTrain==1], axis=0)
meanNotSpam = np.mean(xTrain[yTrain==0], axis=0)
stdSpam = np.std(xTrain[yTrain==1], axis=0)
stdNotSpam = np.std(xTrain[yTrain==0], axis=0)

#Compute the gaussian.
preds = []
for i in range(xTrain.shape[0]):
    probSpam = priorSpam
    probNotSpam = priorNotSpam
    #Sum the likelihood of each feature with the prior for each sample.
    for j in range(xTrain.shape[1]):
        probSpam += natLog(xTrain[i][j], meanSpam[j], stdSpam[j])
        probNotSpam += natLog(xTrain[i][j], meanNotSpam[j], stdNotSpam[j])

    
    #Classify the result
    if probSpam > probNotSpam:
        preds.append(1)
    else:
        preds.append(0)

#Determine accuracy and gather confusion matrix statistics. 
truePos=0
trueNeg=0
falsePos=0
falseNeg=0
for i in range(yTest.shape[0]):
    if preds[i] == 1:
        if yTest[i] == 1:
            truePos += 1
        else:
            falsePos += 1
    else:
        if yTest[i] == 1:
            falseNeg += 1
        else:
            trueNeg += 1
            
print("Training and testing time: %.4f s" % (time.time()-tTrain))
accuracy = (truePos + trueNeg) / (truePos + falsePos + trueNeg + falseNeg)
precision = truePos / (truePos + falsePos)
recall = truePos / (truePos + falseNeg)

print("Accuracy: %.4f Precision: %.4f Recall: %.4f" % (accuracy, precision, recall))
print("Confusion Matrix")
print(confusion_matrix(yTest, preds))
print("Total runtime: %.4f" % (time.time()-tStart))      