print(__doc__)



import json
from sklearn.datasets import load_breast_cancer
import numpy as np
import copy

with open('test(length164).json') as f:
   cellData = json.load(f)





   def createdNestedArrayOfFeatures(numFeature):

    firstArray = []
    secondArray = []
    i = 0
    for cellIndex in range(len(cellData)):   #for every name
         for windowIndex in cellData["Cell"+ str(cellIndex + 1)]:    #for every window in cellData
          for value in cellData["Cell" + str(cellIndex + 1)][windowIndex][numFeature]:
           if value is not None:
            firstArray.append(value)
            if len(firstArray) >= len(cellData["Cell" + str(cellIndex + 1)][windowIndex][numFeature]):
                secondArray.insert(i, copy.copy(firstArray))
                firstArray*= 0
                i = i + 1;
                break

    averagedArray = np.array(secondArray)
    return np.mean(averagedArray, axis = 0)

featureArray1 = createdNestedArrayOfFeatures("Feature1")
featureArray2 = createdNestedArrayOfFeatures("Feature2")
featureArray3 = createdNestedArrayOfFeatures("Feature3")

print(featureArray1)
