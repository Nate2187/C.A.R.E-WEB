from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug import secure_filename
from sklearn.externals import joblib
import predict
import CARE_part2
import os


IMAGE_FOLDER = os.path.join('static', 'graphs')

app = Flask(__name__)

#Set the upload Folder to image Folder
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


@app.route('/')
def index():
     clf = joblib.load('pythonScripts/svc_pca_mod.pkl')
     return render_template('home.html')

#This is the upload section of code
@app.route('/uploaded', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST': #Checks if the post method was sent
      f = request.files['file'] #f gets the fiels that were sent
      f.save(secure_filename(f.filename)) #save f
      timeOfPred = 0    #TODO add input on screen for time
      makePrediction(f.filename, timeOfPred) #'inputTestData.json'
      return displayResult()

#Hook to display prediction result in html
predictionResultsFromData = [15, 14, 15, 15.1]

#This displays the images
@app.route('/<predictionResults>')
def displayResult(predictionResults):
    full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'newFigure1.png')
    full_filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], 'accelGraph.png')
    timeInput = request.form['userTimeInput']
    return render_template('displayResult.html',user_images1 =full_filepath1, user_images = full_filepath, predictionResults = predictionResultsFromData[-1])

def makePrediction(inputFileName, timeOfPred):
    modelMaker = CARE_part2.CARE_part2('testWithZeros.json', 3, inputFileName)
    prediction = predict.predict(modelMaker)

    inputVals = modelMaker.getInputArray()
    #input data set with predicted data added at end
    x_vals = inputVals[0] + [inputVals[0][len(inputVals[0]) - 1] + timeOfPred] #add the latest time to timeOfPred when appending
    y_vals = inputVals[(len(inputVals) - 1)] + [prediction]
    #values = predict.splitPred(prediction)
    predict.saveGraph(inputVals[2], prediction)

if __name__ == '__main__':
    app.run(debug = True)
