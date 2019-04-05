import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def predict(fileName):
    with open(fileName) as f:
        cellData = json.load(f)

    #model = pickle.load(open('.\\pythonScripts\\svc_pca_mod.pkl', "rb"))
    model = pickle.load(open('.\\pythonScripts\\forest_pca_mod.pkl', "rb"))

    cellList = getData(cellData)
    prediction = model.predict(cellList)
    return prediction

#get passed amount of pca features for cell
def getData(cellData):
    data = []
    for n in cellData:
        data.append(cellData[n])
    data = np.array(data).transpose()
    return data


def saveGraph(test_Y, pred_Y):
    time = populateTime(len(pred_Y))
    plt.scatter(time, test_Y, color='black')
    plt.plot(time, pred_Y, color='blue', linewidth=3)
    plt.title("Predicted Cell Growth Velocity")
    plt.xticks(np.arange(0, 200, step=20))
    plt.yticks(np.arange(-5, 20, step=2))
    plt.xlabel("time")
    plt.ylabel("velocity")

    #my_path
    plt.savefig('./static/graphs/newFigure.png')

def populateTime(time):
    timeArray = []
    for i in range(time):
        timeArray.append(i)
    return timeArray

def splitPred(pred_val):
    with open("inputTestData.json") as f:
        sample_data = json.load(f)

    values = []
    values.append(pred_val[:len(hello) - 1])
    true_val = sample_data["Feature3"]
    values.append(np.array(true_val[1:]).transpose())
    return values


hello = predict('inputTestData.json')
values = splitPred(hello)

saveGraph(values[1], values[0])
