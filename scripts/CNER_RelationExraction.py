import pandas as pd
import numpy as np
import re

from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score

import pickle

'''
This class acts as a classifier for relation between entities. It has methods to train and test the relationship classifier model.
Below are the details of the methods of the class.

fit(self, data)-
This methods accepts data as a parameter, uses relation_type as the column to identify relationship between entities. It then trains a logistic regression model for the entity, and then saves the model
in a path defined by model_path.

predict_proba(self,data):
Using the model, classifies the words in given data and returns a list of probabilities.

predict(self,data):
Using the model, classifies the relations in given data and returns a relationship type.

test_data(self,data):
Tests the model, classifies the relations in given data and returns accuracy, f1 score and confusion matrix results.
'''

class RelationExtraction():
    def __init__(self, model_path):
        self.model_path = model_path
        
    def fit(self, data):
        X = np.vstack(list(data["relation_vector"]))                                
        y = data["relation_type"]  
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr',max_iter=1000).fit(X, y)
        
        pickle.dump(clf,open(self.model_path,'wb'))
    
    def predict_proba(self,data):
        X = np.vstack(list(data["relation_vector"]))                                
        
        clf = pickle.load(open(self.model_path,'rb'))
        
        return clf.predict_proba(X)
    
    def predict(self,data):
        X = np.vstack(list(data["relation_vector"]))                                
        
        clf = pickle.load(open(self.model_path,'rb'))
        
        return clf.predict(X)
        
    def test_data(self,data):
        X = np.vstack(list(data["relation_vector"]))                                
        y = data["relation_type"]    
        
        clf = pickle.load(open(self.model_path,'rb'))
        
        y_pred = clf.predict(X)
        
        accuracy = clf.score(X,y)
        print("Score on test data:",accuracy)
        print(confusion_matrix(y,y_pred))
        f1_score_test = f1_score(y, y_pred,average='macro')
        print("F1 Score:",f1_score_test)

        precision = precision_score(y, y_pred,average='macro')
        recall = recall_score(y,y_pred,average='macro')

        print("Precision:",precision)
        print("Recall:",recall)

        return [accuracy,f1_score_test,precision,recall]

        