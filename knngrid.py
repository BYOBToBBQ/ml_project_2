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

param_grid = {'sim_options': {'name': ['pearson_baseline'],
                              'shrinkage': [90,100,110]}
              }

gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(surp)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

print(gs.cv_results)
