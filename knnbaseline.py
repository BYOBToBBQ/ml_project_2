import numpy as np
import pandas as pd
import random
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import BaselineOnly
from surprise import CoClustering
from surprise.model_selection import KFold
from surprise import accuracy
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from joblib import Parallel, delayed
import multiprocessing
import time

def get_users(line):
    row, col = line.split("_")
    row = row.replace("r", "")
    return int(row)

def get_items(line):
    row, col = line.split("_")
    col = col.replace("c", "")
    return int(col)

data = pd.read_csv('data_train.csv')

data['userID'] = data['Id'].apply(get_users)
data['itemID'] = data['Id'].apply(get_items)
data = data.drop('Id', axis=1)
data = data.rename(columns={'Prediction':'rating'})[['userID','itemID','rating']]

reader = Reader(rating_scale=(1, 5))
surp = Dataset.load_from_df(data, reader)

sim_options = {'name': 'pearson_baseline'}

algo = KNNBaseline(sim_options = sim_options)

trainset = surp.build_full_trainset()

algo.fit(trainset)

test = pd.read_csv('examples_sample_submission.csv')
test['userID'] = test['Id'].apply(get_users)
test['itemID'] = test['Id'].apply(get_items)

preds = test.apply(lambda row: round(algo.predict(row.userID, row.itemID).est) , axis=1)
test['Prediction'] = preds
test = test.drop(['userID','itemID'], axis=1)


test.to_csv('subKNNBaselinePearson.csv', index=False)

