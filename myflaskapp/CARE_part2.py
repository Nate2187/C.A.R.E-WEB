# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:58:24 2019

@author: Matthew
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

class CARE_part2:
    
    def appendNewCell(inputData):
        return []

    def __init__(self, inputFileStr, numFeat):
        #original data set
        with open('testWithZeros.json') as f:
            self.cellData = json.load(f)
        #input dataset
        with open(inputFileStr) as f1:
            self.inputData = json.load(f1)
        
        self.numFeat = numFeat
        self.finalCellData = appendNewCell(self.inputData)
        #self.finalCellData = self.cellData.append(self.inputData) #trying to combine the cellData and the inputData
        #get pca of cellData
        #get pca of above (this will be used for the training)
        
        
    def generateModel():
        print("placeholder")
        
    #get passed amount of pca features for cell
    def getPCACells():   
        pcaVals = []
        for n in range(numFeat):
            pcaVals.append(getPCACellFeature(n))
        return pcaVals
    
    #gets pca value in cell for passed feature
    def getPCACellFeature(fNum):
        pca = PCA(n_components=1) #make it one dimensional
        cellVal = []    #the pca for all features in cell
        for c in cellData:  #c is the cell name
            cellVal.append(getPCAWinFeature(c, fNum))
        pca.fit(np.array(cellVal).transpose())
        cellFeat = pca.transform(np.array(cellVal).transpose())
        
        return cellFeat.transpose()[0].tolist()
    
        #will probably need to scale
        #gets the pca for window and feature provided
    def getPCAWinFeature(cellName, fNum):
        pca = PCA(n_components=1)
        feature = []
        for w in cellData[cellName]:    #for every window in cellData
            if len(cellData[cellName][w]["Feature" + str(fNum + 1)]) != 0:
             feature.append(cellData[cellName][w]["Feature" + str(fNum + 1)]) #append passed feature to featureArray
        pca.fit(np.array(feature).transpose())  #train the pca for every windows feature
        winFeat = pca.transform(np.array(feature).transpose())  #create the one dimensional array for this window
        
        #return window feature in following format x = [[...]]
        return winFeat.transpose()[0].tolist()
        
hello = CARE_part2('testWithZeros.json', 3)
print(hello.numFeat)