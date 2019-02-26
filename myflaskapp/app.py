from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug import secure_filename
from sklearn.externals import joblib
import os


IMAGE_FOLDER = os.path.join('static', 'graphs')

app = Flask(__name__)

#Set the upload Fodler to image Folder
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
      return displayResult()

#This displays the image
@app.route('/displayResult')
def displayResult():
    full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'unknown.jpg')
    return render_template('displayResult.html', user_images = full_filepath)


if __name__ == '__main__':
    app.run(debug = True)


    #json_ = request.json
    #Import the MODEL
    #Create a dictionary with Key as Cell(#), value as new dictionary
    #subdictionary key as Window(#), value as new subdictionary
    #subdictionary key as Feature(#), value as float
    #enable the upload
