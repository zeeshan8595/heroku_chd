import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import Parallel, delayed
import joblib

app = Flask(__name__)
model = joblib.load(open('et_500_chd_joblib','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction
    if output==0:
        msg1='the patient is not having Coronary Heart Disease'

    else:
        msg1='the patient is having Coronary Heart Disaese and Further investigations adviced'

    return render_template('index.html', prediction_text='The Model Predicts that {}'.format(msg1))



if __name__ == "__main__":
    app.run(debug=True)