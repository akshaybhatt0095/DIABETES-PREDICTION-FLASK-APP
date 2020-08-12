# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the Random Forest CLassifier model
filename = 'classifier.pkl'
model = pickle.load(open(filename, 'rb'))



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = model.predict(data)
        
        return render_template('index.html', prediction_text='Do you have diabetes?  {}'.format(my_prediction))

if __name__ == '__main__':
	app.run(debug=True)