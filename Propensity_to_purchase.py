#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:16:28 2022

@author: alexvacco
"""


#Customer Propensity to Purchase

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import seaborn as sns

#training data 
train = pd.read_csv('training_sample.csv')

#train.describe()
#train.info()
#train.head()

#Correlation check, between website actions

#headmap to view potential correlations

def correlations():
    'this function will creat a heatmap for our correlations and check correlation (r)'
    #using functions 
    corr = train.corr()
    plt.figure(figsize=(16,14))
    sns.heatmap(corr, vmax = 0.5, center = 0, square = True, linewidths = 2, cmap = 'Blues')
    plt.savefig('heatmap_customers.png')
    plt.show
    
    #highest correlations between orders and basket interactions, sign in and delivery dates
    
    #list correlations r
    train.corr()['ordered']
    
    #obviously outliers drop columns with low correlation
    
predictors = train.drop(['ordered','UserID', 'device_mobile'], axis = 1)
targets = train.ordered

X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size = .3)
#print('Predictor - training : ' + X_train.shape + ' Predictor - testing : ' + X_test.shape)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

sklearn.metrics.confusion_matrix(y_test,predictions)

#load previous days visitors
yesterday_prospects = pd.read_csv('testing_sample.csv')

userids = yesterday_prospects.UserID
yesterday_prospects = yesterday_prospects.drop(['ordered','UserID','device_mobile'], axis = 1)
print(yesterday_prospects.head(10))

#shape
yesterday_prospects.shape

yesterday_prospects['propensity'] = classifier.predict_proba(yesterday_prospects)[:,1]

print(yesterday_prospects.head())
pd.DataFrame(userids)
results = pd.concat([userids, yesterday_prospects], axis = 1)

print(results.head(30))

results.to_csv('propensity_results.csv')