from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug import secure_filename
from sklearn.externals import joblib
import predict
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
      makePrediction(f.filename) #'inputTestData.json'
      return displayResult()

predictionResultsFromData = [15, 14, 15, 15.1]

#This displays the images
@app.route('/<predictionResults>')
def displayResult(predictionResults):
    full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'newFigure.png')
    return render_template('displayResult.html', user_images = full_filepath, predictionResults = predictionResultsFromData[-1])

def makePrediction(fileName):
    prediction = predict.predict(fileName)
    values = predict.splitPred(prediction)
    predict.saveGraph(values[1], values[0])

if __name__ == '__main__':
    app.run(debug = True)
