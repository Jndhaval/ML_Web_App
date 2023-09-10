# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('D:/Dhaval/Programming/Anaconda/Spyder/ml_web_app/trained_model.sav', 'rb'))

i_d = (71,0,0,112,149,0,1,125,0,1.6,1,0,2)

# changing the input data to a numpy array
i_d_n = np.asarray(i_d)

# reshape the numpy array as we are predicting for only one instane
i_d_rp = i_d_n.reshape(1,-1)

prediction = loaded_model.predict(i_d_rp)
print(prediction)

if(prediction[0]==0):
  print("the Person Does Not Have Heart Disease")
else:
  print("The Person Has Heart Disease")