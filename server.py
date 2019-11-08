import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from Model_run import Model_run
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)# Load the model
model = pickle.load(open('model/model2019-10-27.pkl','rb'))
mr=Model_run()

@app.route('/api/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    if float(data['Issue time'])< float(1200):
        tod='AM'
    else:
        tod='PM'
    convert_dict = {
        "Ticket number": int(data['Ticket number']),
        "Issue time": float(data['Issue time']),
        "RP State Plate": str(data['RP State Plate']),
        "Plate Expiry Date": int(data['Plate Expiry Date']),
        "Body Style": str(data['Body Style']),
        "Color": str(data['Color']),
        "Location": str(data['Location']),
        "Route": str(data['Route']),
        "Agency": int(data['Agency']),
        "Violation code": int(data['Violation code']),
        "Violation Description": int(data['Violation Description']),
        "Fine amount": int(data['Fine amount']),
        "Latitude": float(data['Latitude']),
        "Longitude": float(data['Longitude']),
        "Time of Day": tod
    }
    train = pd.DataFrame.from_dict(convert_dict, orient='index')
    transpose_df = train.transpose()
    convert={
        "Ticket number":int,
        "Issue time":float,
        "Plate Expiry Date":float,
        "Agency":int,
        "Violation code":int,
        "Violation Description":int,
        "Fine amount":float,
        "Latitude": float,
        "Longitude": float
    }
    transpose_df=transpose_df.astype(convert)
    categorical_feature_mask = transpose_df.dtypes==object
    categorical_cols = transpose_df.columns[categorical_feature_mask].tolist()
    le = LabelEncoder()
    df_encode = transpose_df
    df_encode[categorical_cols] = df_encode[categorical_cols].apply(lambda col: le.fit_transform(col))
    prediction = model.predict(df_encode)    # Take the first value of prediction
    output = prediction
    output_proba=model.predict_proba(df_encode)
    return jsonify(str(output),str(output_proba))

if __name__ == '__main__':
    app.run(port=5000, debug=True)