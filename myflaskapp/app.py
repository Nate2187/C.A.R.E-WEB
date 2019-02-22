from flask import Flask, render_template, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
json_ = request.json


#Import the MODEL


#Create a dictionary with Key as Cell(#), value as new dictionary
#subdictionary key as Window(#), value as new subdictionary
#subdictionary key as Feature(#), value as float







    return render_template('home.html')

@app.route('/displayResult')
def displayResult():


    return render_template('displayResult.html')




if __name__ == '__main__':
    app.run(debug = True)
