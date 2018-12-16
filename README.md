# ML Recommender Systems Project 2

## Installations

In order to execute our code you will need to install the python libraries Surprise, Numpy, Pandas and Scikit. Instructions for installation are available on their respective websites.  
  
Suprise is a python scikit building and analyzing recommender systems.  
- pip install scikit-surprise
- conda install -c conda-forge scikit-surprise

## Files

- helpers: Various helpers functions to move from the crowdai file format to the Surprise one as well as basic recommender systems implementations.
- exploratory: Provides an exploratory data analysis of the training set.
- tuning_evaluation: Provides the whole pipeline we used in order to tune our individual models as well as the model blending algorithm. For each model evaluation metrics are given.
- run: Fit every model on the whole dataset then produces the model blending predictions for the crowdai challenge.
