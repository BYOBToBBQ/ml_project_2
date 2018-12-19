# ML Recommender Systems Project 2

## Installations

In order to execute our code you will need to install the python libraries Surprise, Numpy, Pandas and Scikit. Instructions for installation are available on their respective websites.  
  
Suprise is a python scikit building and analyzing recommender systems.  
- pip install scikit-surprise
- conda install -c conda-forge scikit-surprise

## Files

- helpers: Various helpers functions to convert the crowdai file format to the Surprise one as well as basic recommender systems implementations.  
  
- exploratory: Provides an exploratory data analysis of the training set.  
  
- tuning_evaluation: Provides the whole pipeline we used in order to tune our individual models as well as the model blending algorithm. For each model evaluation metrics are given. 
    - The training set is first split into two parts: a model training set (80%) and a blender training        set (20%)
    - On the model training set we tune hyperparameters based on a grid search approach. We pick              parameters combinations based on their performance using a 3 Fold Cross Validation (RMSE objective       function) procedure. Then we retrain the tuned models on the whole model training set.
    - On the blender training set we first evaluate the performance of our individual models and store         the ratings predictions. Therefore this set also acts as a validation set. Then we split the             blender training set into two parts: a train set (15%) and a test set (5%).
        - On the train set we perform a ridge regression based on individual predictions as covariates.            Therefore we modelise the true prediction as a penalised linear combination of predictions. The           regularisation parameter is estimated through grid search and 3 Fold Cross Validation.
        - On the test set we evaluate the performance of our model blending algorithm.  
  
- run: In the tuning_evaluation notebook we found appropriate hyperparameters for our invidivual models as well as the model blending algorithm. Now we fit every tuned model on the whole training set then we output predictions as a linear combination of predictions.
