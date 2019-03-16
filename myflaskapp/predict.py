import pickle
import json
import numpy as np

def predict(fileName):
    with open(fileName) as f:
        cellData = json.load(f)
    
    model = pickle.load(open('C:\\Users\\Matthew\\CARE Project\\CARE_Part_2\\svc_pca_mod.pkl', "rb"))
    #prediction = model.predict(cellData)
    return cellData

hello = predict('inputTestData.json')

def getPercentFeatureData(train, featureData, percent):
    combinedFeatures = []
    for feature in featureData:
        if train:
            combinedFeatures.append(feature[:int(len(feature) * percent)])
        else:
            combinedFeatures.append(feature[int(len(feature) * percent + 1):int(len(feature) - 1)])
    return combinedFeatures