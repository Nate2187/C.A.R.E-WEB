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
        else:
            with open(inputFileAddress) as f1:
                self.inputData = json.load(f1)
            self.finalCellData = self.appendNewCell(self.cellData, self.inputData) #create new data set that includes user input
            
        self.TRAIN_PERCENT = .8
        self.numFeat = numFeat
        self.finalListedData = [[]]
        
        
    def appendNewCell(self, baseData, inputData):
        cellName = "Cell" + str(len(self.cellData) + 1)
        baseDataClone = copy.deepcopy(baseData)
        if inputData.get('Window1') is None:
            baseDataClone[cellName] = {'Window1':inputData}
        else:
            baseDataClone[cellName] = inputData
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
        cellMerge = CellDataMerge()
        cell_data = cellMerge.getCombinedCells(self.finalCellData, 3)
        self.finalListedData = cell_data
        
        #get training input for pca data
        train_XList = self.getPercentFeatureData(True, cell_data, self.TRAIN_PERCENT)   #keep for record
        train_X = np.array(train_XList).transpose()
        
        #get training target data for pca data
        target = cell_data[len(cell_data) - 1]
        train_Y = target[1:int(len(target) * self.TRAIN_PERCENT) + 1] #expected output
        
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
        
    def getPercentFeatureData(self, train, featureData, percent):
        combinedFeatures = []
        for feature in featureData:
            if train:
                combinedFeatures.append(feature[:int(len(feature) * percent)])
            else:
                combinedFeatures.append(feature[int(len(feature) * percent + 1):int(len(feature) - 1)])
        return combinedFeatures

   # def get

class PCAImplement:
    #get passed amount of pca features for cell
    def getPCACells(self, inputData, fNum):   
        pcaVals = []
        for n in range(fNum):
            pcaVals.append(self.getPCACellFeature(inputData, n))
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
    
class CellDataMerge:
    
    def getCombinedCells(self, inputData, numFeats):
        combinedCellData = []
        for f in range(numFeats):
            combinedCellData.append(self.getCombCellFeat(inputData, f))
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
            feature += self.getFeat(winObj, fNum)
        return feature
                
    def getFeat(self, winObj, fNum):
        featName = "Feature" + str(fNum + 1)
        #check if there are any values in the feature
        if len(winObj[featName]) != 0:
            return winObj[featName] #append passed feature to featureArray
        else:
            return []
            
hello = CARE_part2('testWithZeros.json', 3, 'inputTestData.json')
hello.generateModel(2)