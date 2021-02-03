import pandas as pd
from sklearn.neighbors import NearestNeighbors

traindata = pd.read_csv('data/Train.csv')
testdata = pd.read_csv('data/Test.csv')
submitdata = pd.read_csv('data/SampleSubmission.csv')

X, y = traindata.Text, traindata.Label

print(X.shape)