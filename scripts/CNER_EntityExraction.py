import pandas as pd
import numpy as np
import re

from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score

import pickle

'''
This class acts as a classifier for an entity. It has methods to train and test the entity classifier model.
Below are the details of the methods.

resolve_continuity(text,words,ignore_letter_list)- 
This method accepts a text, a list of words tagged as yes/no for a given entity, and then combines all the words
which are tagged as positive for the entity. For example in the sentence - "I am suffering from mild headache", if
'mild' and 'headache' are both tagged as positive events, it means these are not separate events, rather words of 
the same event. This method identifies all such cases and returns a list of words after resolving the continuities.

fit(self, data)-
This methods accepts data as a parameter, uses column_name_for_flag argument to identify the column for distinguishing 
between positive and negative cases. It then trains a logistic regression model for the entity, and then saves the model
in a path defined by model_path.

predict_proba(self,data):
Using the model, classifies the words in given data and returns a list of probabilities.

predict(self,data):
Using the model, classifies the words in given data and returns a 0/1 values.

test_data(self,data):
Tests the model, classifies the words in given data and returns accuracy, f1 score and confusion matrix results.
'''

def resolve_continuity(text,words,ignore_letter_list=[]):
    list_of_positions = []
    first_entry = True
    prev_entry = None
    
    for index,entry in words.iterrows():
        if(not first_entry):
            if(entry['entity_flag']==prev_entry['entity_flag']):
                in_between_text = text[prev_entry["end_pos"]:entry["begin_pos"]]
                for letter in ignore_letter_list:
                    in_between_text = re.sub(letter, ' ', in_between_text)
                if(len(in_between_text.strip())==0):
                    prev_entry["end_pos"] = entry["end_pos"]
                    prev_entry["word"] = text[prev_entry["begin_pos"]:prev_entry["end_pos"]]
                    prev_entry["event_probab"] = (prev_entry["event_probab"] + entry["event_probab"])/2
                    prev_entry["timex_probab"] = (prev_entry["timex_probab"] + entry["timex_probab"])/2
                else:
                    list_of_positions.append(prev_entry)
                    prev_entry = entry
            else:
                list_of_positions.append(prev_entry)
                prev_entry = entry
        else:
            prev_entry = entry
            first_entry =False
            
    if(prev_entry is not None):
        list_of_positions.append(prev_entry)
    
    return list_of_positions

class EntityExtraction():
    def __init__(self, model_path,column_name,ignore_letter_list='[ ]',downsample=False,downsample_multiplier=1,bert_layer=0):
        self.model_path = model_path
        self.column_name_for_flag = column_name
        self.ignore_letter_list = ignore_letter_list
        self.downsample = downsample
        self.downsample_multiplier = downsample_multiplier
        self.bert_layer = bert_layer
        
    def fit(self, data):
        pos_data = data[data[self.column_name_for_flag]==1]
        neg_data = data[data[self.column_name_for_flag]==0]
        
        if(self.downsample):
            pos_data_volume = len(pos_data)
            neg_data = neg_data.sample(pos_data_volume*self.downsample_multiplier)
        
        combined_data = pd.concat([pos_data,neg_data])
        
        combined_data['input_vector'] = combined_data.apply(lambda x: x['keyword_vector'][self.bert_layer],axis=1)
        
        X = np.vstack(list(combined_data["input_vector"]))                                
        y = combined_data[self.column_name_for_flag]  
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr',max_iter=1000).fit(X, y)
        pickle.dump(clf,open(self.model_path,'wb'))
    
    def predict_proba(self,data):
        data['input_vector'] = data.apply(lambda x: x['keyword_vector'][self.bert_layer],axis=1)
        
        X = np.vstack(list(data["input_vector"]))                                
        
        clf = pickle.load(open(self.model_path,'rb'))
        
        return clf.predict_proba(X)
    
    def predict(self,data):
        data['input_vector'] = data.apply(lambda x: x['keyword_vector'][self.bert_layer],axis=1)
        
        X = np.vstack(list(data["input_vector"]))                                
        
        clf = pickle.load(open(self.model_path,'rb'))
        
        return clf.predict(X)
        
    def test_data(self,data):
        data['input_vector'] = data.apply(lambda x: x['keyword_vector'][self.bert_layer],axis=1)
        
        X = np.vstack(list(data["input_vector"]))                                                                
        y = data[self.column_name_for_flag]   
        
        clf = pickle.load(open(self.model_path,'rb'))
        y_pred = clf.predict(X)
        
        accuracy = clf.score(X,y)
        print("Score on test data:",accuracy)
        print(confusion_matrix(y,y_pred))
        f1_score_test = f1_score(y, y_pred)
        print("F1 Score:",f1_score_test)

        precision = precision_score(y, y_pred)
        recall = recall_score(y,y_pred)

        print("Precision:",precision)
        print("Recall:",recall)

        return [accuracy,f1_score_test,precision,recall]

        