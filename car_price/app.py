from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np 
import json 

#loading the model
gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def car_price():
    data = {}
    if request.form:
        result = request.form
        data['form'] = result
        year_model = float(result['year_model'])
        mileage = float(result['mileage'])
        mark = result['mark']
        fiscal_power = float(result['fiscal_power'])
        fuel_type = result['fuel_type']
        user_input = {'year_model':year_model, 'mileage':mileage, 'fiscal_power':fiscal_power, 'fuel_type':fuel_type, 'mark':mark}
        
        #convert user input to one hot vector
        enc_input = np.zeros(62)
        year_mean = 2006.17
        year_std = 8.2
        mileage_mean = 127561.48
        mileage_std = 91516.96
        fiscal_mean = 7.5
        fiscal_std = 2.3
        #list of columns 
        cols = ['year_model', 'mileage', 'fiscal_power', 'fuel_type_Diesel',
                'fuel_type_Electriq', 'fuel_type_Ess', 'mark_Ac', 'mark_Alfa Romeo',
                'mark_Audi', 'mark_Autres', 'mark_BMW', 'mark_BYD', 'mark_Cadillac',
                'mark_Changh', 'mark_Chery', 'mark_Chevrolet', 'mark_Chrysl',
                'mark_Citroen', 'mark_Daci', 'mark_Daewoo', 'mark_Daihats', 'mark_Dodg',
                'mark_Fiat', 'mark_Ford', 'mark_Geely', 'mark_Hond', 'mark_Humm',
                'mark_Hyundai', 'mark_Infiniti', 'mark_Isuz', 'mark_Iveco', 'mark_Jag',
                'mark_Jeep', 'mark_Ki', 'mark_Land Rov', 'mark_Lexus', 'mark_Lincoln',
                'mark_Nissan', 'mark_Opel', 'mark_Peugeot', 'mark_Pontiac',
                'mark_Porsch', 'mark_Renault', 'mark_Rov', 'mark_Seat', 'mark_Skod',
                'mark_Smart', 'mark_Ssangyong', 'mark_Sub', 'mark_Suzuki', 'mark_Toyot',
                'mark_Volkswagen', 'mark_Volvo', 'mark_Zoty', 'mark_cedes-Benz',
                'mark_hind', 'mark_itsubishi', 'mark_lanci', 'mark_mini', 'mark_serati',
                'mark_sey Ferguson', 'mark_zd']
        #set the numerical features
        enc_input[0] = (user_input['year_model']-year_mean)/year_std
        enc_input[1] = (user_input['mileage']-mileage_mean)/mileage_std
        enc_input[2] = (user_input['fiscal_power']-fiscal_mean)/fiscal_std
        #convert the mark to match the column name
        mark_col = 'mark_'+user_input['mark']
        #find the index of mark_col
        mark_ind = cols.index(mark_col)
        #set the corresponding entry to be 1 
        enc_input[mark_ind] = 1
        #convert the fuel type to match the column name
        fuel_col = 'fuel_type_'+user_input['fuel_type']
        #find the index of fuel_col
        fuel_ind = cols.index(fuel_col)
        #set the corresponding entry to be 1
        enc_input[fuel_ind] = 1 

        #get prediction
        price = gbr.predict([enc_input])[0]
        price = round(price,2)
        data['prediction'] = 'The predicted price of the car is {}'.format(price)
    return render_template('index.html', data = data)

 #script initialization
if __name__ == '__main__':
    app.run(debug = True)   



