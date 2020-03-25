from CNER_RelationExraction import RelationExtraction
from CNER_DischargeNote import DischargeNote

import os
import xml.etree.ElementTree as ET
import pandas as pd
from progressbar import ProgressBar
import random

'''

This file is to train and test the relationships.

It first goes through all the files, devides the list of files in 4:1 ratio. 

Then it uses 80% of the data to train the relation extraction model and then remaining 20% data to test the model.

'''
print("File reading in progress...")

relation_list = []
count = 0
pbar = ProgressBar()
all_files = []

for file in pbar(os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/all_data/")):
    try:
        if file.endswith(".xml"):
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/all_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = DischargeNote(root,file,baseline=False)
            discharge_note.process_note()
            relation_list.extend(discharge_note.relation_list)
            all_files.append(file)
    except:
        print("Error in processing file:",file)


relation_list_df = pd.DataFrame(relation_list)

random.shuffle(all_files)

test_file_size = int(len(all_files)/5)
result_list = []

test_files = all_files[0:test_file_size]
train_files = list(set(all_files) - set(test_files))
train_data_df = relation_list_df[relation_list_df['file_name'].isin(train_files)]
test_data_df = relation_list_df[relation_list_df['file_name'].isin(test_files)]
print("Length of training data:", len(train_data_df))

relation_extraction = RelationExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/relation.pkl")
        
relation_extraction.fit(train_data_df)
        
relation_extraction.test_data(test_data_df)
