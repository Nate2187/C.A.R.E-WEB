import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import CARE_part2

def predict(fileName):
    modelMaker = CARE_part2.CARE_part2('testWithZeros.json', 3, fileName)

    #model = pickle.load(open('.\\pythonScripts\\svc_pca_mod.pkl', "rb"))
    #model = pickle.load(open('.\\pythonScripts\\forest_pca_mod.pkl', "rb"))


    inputArray = getInputArray(modelMaker.inputData, len(modelMaker.inputData))
    trueInput = getTrueInput(inputArray)
    model = modelMaker.generateModel(2)
    prediction = model.predict(trueInput)
    return prediction[0]

#return an array of the input data
def getInputArray(cellData, numFeats):
    data = []
    converter = CARE_part2.CellDataMerge()
    for f in range(numFeats):
        data.append(converter.getFeat(cellData, f))
    return data

def getTrueInput(inputArray):
    inputVal = []
    inputs = []
    for f in inputArray:
        inputs += f[len(f) - 1:]
    inputVal.append(inputs)
    return inputVal

#def saveGraph(test_Y, pred_Y):
#    time = populateTime(len(pred_Y))
#    plt.scatter(time, test_Y, color='black')
#    plt.plot(time, pred_Y, color='blue', linewidth=3)
#    plt.title("Predicted Cell Growth Velocity")
#    plt.xticks(np.arange(0, 200, step=20))
#    plt.yticks(np.arange(-5, 20, step=2))
#    plt.xlabel("time")
#    plt.ylabel("velocity")
#
#    #my_path
#    plt.savefig('./static/graphs/newFigure.png')

#def populateTime(time):
#    timeArray = []
#    for i in range(time):
#        timeArray.append(i)
#    return timeArray

def saveGraph(arrayOfInput1, arrayOfOutput1):
    arrayOfInput = [20,25,23,29]
    arrayOfOutput= [33,35,34, 34]
    arrayOfOutput= [arrayOfInput[-1]] + arrayOfOutput #
    #plt.scatter(time, test_Y, color='black')
    arrayHolder = arrayOfInput+ arrayOfOutput
    time1 = populateTime(0,len(arrayOfInput)) #[1,2,3,4]
    time = populateTime(len(arrayOfInput)-1, len(arrayOfInput) + len(arrayOfOutput)-1)#[4,5,6, 7]


    plt.plot(time1, arrayOfInput, color='blue', linewidth=3)
    plt.plot(time, arrayOfOutput, color='green', linewidth=3)
    plt.title("Predicted Cell Growth Velocity")
    plt.axvline(x= len(arrayOfInput)-1, color= 'red', linewidth=5)
    plt.xticks(np.arange(0, len(arrayHolder), step=1))
    plt.yticks(np.arange(10, 40, step=2))
    plt.xlabel("Time")
    plt.ylabel("Velocity")

    #my_path
    plt.savefig('./static/graphs/newFigure1.png')

def populateTime(start,time):
    timeArray = []
    for i in range(start, time):
        timeArray.append(i)
    return timeArray

def splitPred(pred_val):
    #test data to compare predicted value with based off sample from original data set
    with open("inputTestData.json") as f:
        sample_data = json.load(f)

    values = []
    values.append(pred_val[:len(pred_val) - 1])
    true_val = sample_data["Feature3"]
    values.append(np.array(true_val[1:]).transpose())
    return values


predicted_array = predict('inputTestData.json')
print(predicted_array)
#values = splitPred(predicted_array)

saveGraph(values[1], values[0])
