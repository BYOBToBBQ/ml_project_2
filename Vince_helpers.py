import numpy as np
import pandas as pd
import random
import math

#Helpers functions to go from the crowdai format to the surprise one

def get_users(line):
    '''Get the userID'''
    row, col = line.split("_")
    row = row.replace("r", "")
    return int(row)

def get_items(line):
    '''get the movieID'''
    row, col = line.split("_")
    col = col.replace("c", "")
    return int(col)

def to_surprise(data):
    '''Move the dataframe from the Id/rating format to the userID/itemID/rating format'''
    data['userID'] = data['Id'].apply(get_users)
    data['itemID'] = data['Id'].apply(get_items)
    data = data.drop('Id', axis=1)
    data = data.rename(columns={'Prediction':'rating'})[['userID','itemID','rating']]
    return data

#Basic solutions models

def global_mean(df):
    '''Return the global dataset mean'''
    return df.rating.mean()

def user_mean(df):
    '''Return a serie where each user is associated with his mean'''
    return df.groupby('userID').rating.mean()

def movie_mean(df):
    '''Return a serie where each movie is associated with its mean'''
    return df.groupby('itemID').rating.mean()

def predict_user(uid, users, mean):
    '''Predict user mean based on the serie returned by user_mean'''
    return users.get(uid, mean)

def predict_movie(iid, movies, mean):
    '''Predict movie mean based on the serie returned by movie_mean'''
    return movies.get(iid, mean)