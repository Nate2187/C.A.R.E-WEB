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
import pickle
import copy

class CARE_part2:
    
    def __init__(self, dataSetAddress, numFeat, inputFileAddress = None):
        #original data set
        with open(dataSetAddress) as f:
            self.cellData = json.load(f)
        #input dataset
        if inputFileAddress is None:
            self.finalCellData = self.cellData
            self.inputData = []
            self.finalListedData = [[]]
        else:
            with open(inputFileAddress) as f1:
                self.inputData = json.load(f1)
            self.finalCellData = self.appendNewCell(self.cellData, self.inputData) #create new data set that includes user input
            cellMerge = CellDataMerge()
            self.finalListedData = cellMerge.getCombinedCells(self.finalCellData, numFeat - 1)
            #pcaMerge = PCAImplement()
            #self.finalListedData = pcaMerge.getPCACells(self.finalCellData, numFeat - 1)
            
        self.numFeat = numFeat
        
        
    #TODO: may need to change if input is standardized to cell-window-feature
    def appendNewCell(self, baseData, inputData):
        cellName = "Cell" + str(len(self.cellData) + 1)
        baseDataClone = copy.deepcopy(baseData)
        if inputData.get('Cell1') is None:
            if inputData.get('Window1') is None:
               baseDataClone[cellName] = {'Window1':inputData}
        else:
            baseDataClone[cellName] = inputData['Cell1']
        return baseDataClone
        
    #in case saving the model to a file is preferable 
    def saveModel(self, alg, saveFileAddress):
        #create model
        model = self.generateModel(alg)
        
        #export the AI model
        pickle.dump(model, open(saveFileAddress,"wb"))
    
    def generateModel(self, alg):
        #pcaImp = PCAImplement()
        #cell_data = pcaImp.getPCACells(self.finalCellData, 3) #reassemble data using chosen method
        
        #get training input for pca data
        train_XList = self.getXData(self.finalListedData)   #keep for record
        train_X = np.array(train_XList).transpose()
        
        #get training target data for pca data
        target = self.finalListedData[len(self.finalListedData) - 1]
        train_Y = target[1:int(len(target))] #expected output
        
        #choose learning algorithm to use
        if alg == 0:
            # support vector machine
            CARE_alg = svm.SVR()
        elif alg == 1:
            # linear regression
            CARE_alg = linear_model.LinearRegression()
        elif alg == 2:
            CARE_alg = RandomForestRegressor()
        
        # Train the model using the training sets
        CARE_model = CARE_alg.fit(train_X, train_Y)
        
        return CARE_model
        
    def getXData(self, featureData):
        combinedFeatures = []
        for feature in featureData:
            combinedFeatures.append(feature[:int(len(feature) - 1)])
        return combinedFeatures

    #return a 2D array of the input data
    def getInputArray(self):
        data = []
        converter = CellDataMerge()
        data = converter.getCombinedCells(self.inputData, self.numFeat - 1)
        return data
    
    #returns the last input set from the input array
    def getTrueInput(self):
        inputVal = []
        inputs = []
        inputArray = self.getInputArray()
        for f in inputArray:
            inputs += f[len(f) - 1:]
        inputVal.append(inputs)
        return inputVal
    

class PCAImplement:
    def __init__(self):
        self.timeArray = []
        
    #get passed amount of pca features for cell
    def getPCACells(self, inputData, fNum):   
        pcaVals = []
        for n in range(fNum):
            pcaVals.append(self.getPCACellFeature(inputData, n))
        self.updateTimeArray(len(pcaVals[0]))
        pcaVals = [self.timeArray] + pcaVals
        return pcaVals
    
    #gets pca value in cell for passed feature
    def getPCACellFeature(self, inputData, fNum):
        pca = PCA(n_components=1) #make it one dimensional
        cellVal = []    #the pca for all features in cell
        for c in inputData:  #c is the cell name
            cellVal.append(self.getPCAWinFeature(inputData, c, fNum))
        pca.fit(np.array(cellVal).transpose())
        cellFeat = pca.transform(np.array(cellVal).transpose())
        
        return cellFeat.transpose()[0].tolist()
    
        #will probably need to scale
        #gets the pca for window and feature provided
    def getPCAWinFeature(self, inputData, cellName, fNum):
        pca = PCA(n_components=1)
        feature = []
        for w in inputData[cellName]:    #for every window in inputData
            if len(inputData[cellName][w]["Feature" + str(fNum + 1)]) != 0:
                feature.append(inputData[cellName][w]["Feature" + str(fNum + 1)]) #append passed feature to featureArray
        pca.fit(np.array(feature).transpose())  #train the pca for every windows feature
        winFeat = pca.transform(np.array(feature).transpose())  #create the one dimensional array for this window
        
        #return window feature in following format x = [[...]]
        return winFeat.transpose()[0].tolist()
    
    def updateTimeArray(self, numTimes):
        for i in range(numTimes):
            self.timeArray.append(i)
    
class CellDataMerge:
    def __init__(self):
        self.timeArray = []
        self.updateTime = True
        
    def getCombinedCells(self, inputData, numFeats):
        combinedCellData = []
        for f in range(numFeats):
            combinedCellData.append(self.getCombCellFeat(inputData, f))
            self.updateTime = False
        combinedCellData = [self.timeArray] + combinedCellData
        return combinedCellData
                    
    def getCombCellFeat(self, inputData, fNum):
        cellVal = []
        for c in inputData:  #c is the cell name
            cObj = inputData[c]
            cellVal += self.getCombWinFeat(cObj, fNum)
        
        return cellVal
            
    def getCombWinFeat(self, cellObj, fNum):
        feature = []
        for w in cellObj:    #for every window in inputData
            winObj = cellObj[w]
            feat = self.getFeat(winObj, fNum)
            self.updateTimeArray(len(feat))
            feature += feat
        return feature
                
    def getFeat(self, winObj, fNum):
        featName = "Feature" + str(fNum + 1)
        #check if there are any values in the feature
        if len(winObj[featName]) != 0:
            return winObj[featName] #append passed feature to featureArray
        else:
            return []
        
    def updateTimeArray(self, numTimes):
        if self.updateTime:
            for i in range(numTimes):
                self.timeArray.append(i)
"""     
#testing code      
hello = CARE_part2('testWithZeros.json', 4, 'inputTestData.json')
hello.generateModel(2)

testing = hello.getInputArray()
"""