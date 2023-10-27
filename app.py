from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
appp = Flask(__name__)
model = pickle.load(open('random_forest_regression_modell.pkl', 'rb'))
@appp.route('/',methods=['GET'])
def Home():
    return render_template('indexx.html')



standard_to = StandardScaler()
@appp.route("/predict", methods=['POST'])

def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        presentprice=float(request.form['presentprice'])
        Kilometers_Driven=int(request.form['Kilometers_Driven'])
        Kilometers_Driven2=np.log(Kilometers_Driven)
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        power = float(request.form['power'])
        Owner_Type=int(request.form['Owner_Type'])
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        Year=2023-Year

        Transmission_Manual=request.form['Transmission_Manual']
        if(Transmission_Manual=='Manual'):
            Transmission_Manual=1
        else:
            Transmission_Manual=0
        prediction=model.predict([[presentprice,Kilometers_Driven2,mileage,engine,power,Owner_Type,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Transmission_Manual]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('indexx.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('indexx.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('indexx.html')

if __name__=="__main__":
    appp.run(debug=True)
