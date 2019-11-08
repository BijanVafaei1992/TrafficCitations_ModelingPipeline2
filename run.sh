#!/bin/sh

# Use this shell script to execute the model. 


echo DOWNLOADING DATA AND MAKING YOUR MODEL READY
python Model_run.py

echo NOW STARTING THE SERVER FOR SERVING CREATED MODEL
python server.py
echo SEND POST REQUEST TO http://localhost:5000/api/predict