import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

from flask import Flask, request, render_template

app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))
col=['lotsize', 'bedrooms', 'bathrms', 'stories', 'driveway', 'recroom','fullbase', 'gashw', 'airco', 'garagepl', 'prefarea']

@app.route('/')
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final= [np.array(int_features, dtype=float)]
    prediction=model.predict(final)
    output= round(prediction[0],2)

    #return render_template('index.html', pred="The Price of your dream house is {}".format(output))
    return render_template('index.html', pred='The price of your dream house is {} USD Only.'.format(output))

if __name__=='__main__':
    app.run(debug=True, port= 2121)
