import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import CARE_part2
import copy

def predict(modelMaker, timeOfPred = None):
    #model = pickle.load(open('.\\pythonScripts\\svc_pca_mod.pkl', "rb"))
    #model = pickle.load(open('.\\pythonScripts\\forest_pca_mod.pkl', "rb"))

    trueInput = modelMaker.getTrueInput()
    model = modelMaker.generateModel(2)
    
    #change the time value to timeOfPred if one is given
    if timeOfPred is not None:
        prediction = multiPredict(trueInput, timeOfPred, model)
    else:
        prediction = model.predict(trueInput)
        
    return prediction

def multiPredict(trueInput, timeOfPred, model):
    predArray = [[],[]]
    #changingIn = copy.deepcopy(trueInput)
    changingIn = [[17, 7, 2.389259e+02, -7.623412e-02]]
    for i in range(timeOfPred):
        changingIn[0][0] = i
        predArray[0].append(trueInput[0][0] + i)
        predArray[1].append(model.predict(changingIn)[0])
    return predArray

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

def saveGraph(arrayOfInput, arrayOfOutput):
    largestNum= arrayOfInput[0]
    smallestNum = arrayOfInput[0]
    arrayOfOutput= [arrayOfInput[-1]] + arrayOfOutput
    #plt.scatter(time, test_Y, color='black')

    arrayHolder = arrayOfInput+ arrayOfOutput
    accelerationPoints= [(arrayHolder[-1]-arrayHolder[0])/(len(arrayHolder))] * len(arrayHolder)

    for i in range(0,len(arrayHolder)):
        if (arrayHolder[i]>largestNum):
            largestNum=arrayHolder[i]
        if (arrayHolder[i]<smallestNum):
            smallestNum=arrayHolder[i]

    time1 = populateTime(0,len(arrayOfInput)) #[1,2,3,4]
    time = populateTime(len(arrayOfInput)-1, len(arrayOfInput) + len(arrayOfOutput)-1)#[4,5,6, 7]
    accTime = populateTime(0, len(arrayHolder))
    plt.figure(1)
    plt.plot(accTime, accelerationPoints,color= 'black')
    plt.savefig('./static/graphs/accelGraph.png')
    plt.figure(2)
    plt.plot(time1, arrayOfInput, color='blue', linewidth=3)
    plt.plot(time, arrayOfOutput, color='green', linewidth=3)
    plt.title("Predicted Cell Growth(Velocity/Time)")
    plt.axvline(x= len(arrayOfInput)-1, color= 'red', linewidth=1)
    plt.xticks(np.arange(0, len(arrayHolder)+1, step=(len(arrayHolder)/10))) ###len(arrayHolder)/20)
    plt.yticks(np.arange(smallestNum, largestNum, step=(largestNum-smallestNum)/20))
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    #my_path
    plt.savefig('./static/graphs/newFigure1.png')
    #Acceleration

def populateTime(start,time):
    timeArray = []
    for i in range(start, time):
        timeArray.append(i)
    return timeArray



#test code
modelMaker = CARE_part2.CARE_part2('testWithZeros.json', 4, 'inputTestData.json')
prediction = predict(modelMaker, 100)


print(prediction)
