from flask import Flask, render_template, jsonify, request
from werkzeug import secure_filename
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():


     clf = joblib.load('pythonScripts/svc_pca_mod.pkl')
     UPLOAD_FOLDER = "myflaskapp/data"
     ALLOWED_EXTENSION = set('html')



     return render_template('home.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return ''




@app.route('/displayResult')
def displayResult():


    return render_template('displayResult.html')




if __name__ == '__main__':
    app.run(debug = True)


    #json_ = request.json
    #Import the MODEL
    #Create a dictionary with Key as Cell(#), value as new dictionary
    #subdictionary key as Window(#), value as new subdictionary
    #subdictionary key as Feature(#), value as float
    #enable the upload
