import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import numpy as np
from nltk import word_tokenize
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences

df=pd.read_csv("summary.csv",encoding = "ISO-8859-1")
columns=df.columns

X=df['ctext']
y=df['text']


#An example showing the data and the summary
print("Text:",X[1])
print("\n")
print("Summary:",y[1])

#Splitting the dataset to train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

def preprocessing():
    #This function  will be used to preprocess the data to a form that can be fed into the neural network
    #Subparts:
    ##1.Tokenizing the data
    ##2.Padding the sentences to a specific length
    ##3.Introducing embeddings( We'll be using the Fasttext embeddings)
    pass

def model():
    #This function defines the structure  of the model , i.e the number of hidden units, no. of layers
    #and the input and output dimensions of each layer.(We'll be using keras)
    pass

def model_accuracy():
    #Function to check how the model works on the test data
    pass

def predict():
    #This function will interact with the user in taking whole texts and will return it's summary as output
    pass
