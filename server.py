import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)# Load the model
model = pickle.load(open('model/model.pkl','rb'))\

@app.route('/api/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(np.array(data))    # Take the first value of prediction
    output = prediction
    return jsonify(output)

@app.route('/api/train',methods=['POST'])
def train():
    pass

if __name__ == '__main__':
    app.run(port=5000, debug=True)