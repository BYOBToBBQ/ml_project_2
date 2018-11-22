
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np



def get_ids():
    """
    gets the idea for which we want to estimate the ratings
    returns ids array (n,2) where, 1rs col is r and 2nd is c
    """
    #fucntion transform r23c890 to 23, 890
    def deal_line(line):
        row, col = line.split('_')
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col)
    
    #determines which ID we need in our model
    ids_text = np.genfromtxt("examples_sample_submission.csv", delimiter=",", skip_header=1, dtype=str, usecols=0)
    
    # parse each line
    ids = [deal_line(line) for line in ids_text]
    
    return ids
