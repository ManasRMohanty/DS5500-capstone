import pandas as pd
import numpy as np
import re

from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pickle


def resolve_continuity(text,words,ignore_letter_list):
    list_of_positions = []
    first_entry = False
    prev_entry = None
    for entry in words:
        if(not first_entry):
            in_between_text = text[prev_entry["end_pos"]:entry["begin_pos"]]
            in_between_text = re.sub(ignore_letter_list, ' ', in_between_text)
            if(len(in_between_text.strip())==0):
                prev_entry["end_pos"] = entry["end_pos"]
                prev_entry["prediction_Probability"] = (prev_entry["prediction_Probability"] + entry["prediction_Probability"])/2
            else:
                list_of_positions.append(prev_entry)
                prev_entry = entry
        else:
            prev_entry = entry
            
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
       
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr').fit(X, y)
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
        
        print("Score on test data:",clf.score(X,y))
        print(confusion_matrix(y,y_pred))
        