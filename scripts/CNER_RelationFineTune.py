from CNER_RelationExtraction import RelationExtraction
from CNER_DischargeNote import DischargeNote

import os
import xml.etree.ElementTree as ET
import pandas as pd
from progressbar import ProgressBar
import random
from CNER_Config import bert_config, data_config
import pickle
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

relation_encoding_list = []
relation_label_list = []

for file in pbar(os.listdir(data_config['train_data_path'])):
    try:
        if file.endswith(".xml"):
            file_name = os.path.join(data_config['train_data_path'], file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = DischargeNote(root,file)
            discharge_note.process_note()
            relation_encoding_list.extend(discharge_note.relation_encoding_list)
            relation_label_list.extend(discharge_note.relation_label_list)
            all_files.append(file)
    except:
        print("Error in processing file:",file)


pickle.dump(relation_encoding_list,open("C:/Users/itsma/Documents/Capstone project/relation_encodings_train.pkl","wb"))
pickle.dump(relation_label_list,open("C:/Users/itsma/Documents/Capstone project/relation_labels_train.pkl","wb"))
