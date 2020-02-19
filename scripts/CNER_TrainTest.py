from CNER_EntityExraction import EntityExtraction
from CNER_DischargeNote import DischargeNote

import os
import xml.etree.ElementTree as ET
import pandas as pd
from progressbar import ProgressBar
import random

'''

This file is to train and test the entity extraction models.

'''
print("File reading in progress...")

word_list = []
count = 0
pbar = ProgressBar()
all_files = []

for file in pbar(os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/all_data/")):
    if file.endswith(".xml"):
        file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/all_data/", file)
        tree = ET.parse(file_name)
        root = tree.getroot()
        discharge_note = DischargeNote(root,file)
        discharge_note.process_note()
        word_list.extend(discharge_note.processed_text)
        all_files.append(file)
        #except:
            #print("Error in processing file:",file)


word_list_df = pd.DataFrame(word_list)
random.shuffle(all_files)

'''
print("File reading for testing in progress...")
word_list_test = []
count = 0
pbar = ProgressBar()

for file in pbar(os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/")):
    if file.endswith(".xml"):
        try:
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = DischargeNote(root)
            discharge_note.process_note()
            word_list_test.extend(discharge_note.processed_text)
        except:
            print("Error in processing file:",file)
'''
test_file_size = int(len(all_files)/5)
result_list = []

for i in range(5):
    test_files = all_files[(i*test_file_size):((i+1)*test_file_size)]
    train_files = list(set(all_files) - set(test_files))
    train_data_df = word_list_df[word_list_df['file_name'].isin(train_files)]
    test_data_df = word_list_df[word_list_df['file_name'].isin(test_files)]
    print(len(train_data_df))
    print("K-fold Iteration:",i)

    for bert_layer in range(13):
        new_dict = {}
        new_dict['K-fold_iteration'] = i
        new_dict['bert_layer'] = bert_layer

        print("Layer:",bert_layer)

        entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/event.pkl","event_flag",bert_layer=bert_layer)
        entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/timex.pkl","timex_flag",downsample_multiplier=4,bert_layer=bert_layer)
        
        entity_extraction.fit(train_data_df)
        entity_extraction_timex.fit(train_data_df)
        
        test_results_event = entity_extraction.test_data(test_data_df)
        test_results_timex = entity_extraction_timex.test_data(test_data_df)

        new_dict['event_model_f1'] = test_results_event[1]
        new_dict['event_model_precision'] = test_results_event[2]
        new_dict['event_model_recall'] = test_results_event[3]

        new_dict['timex_model_f1'] = test_results_timex[1]
        new_dict['timex_model_precision'] = test_results_timex[2]
        new_dict['timex_model_recall'] = test_results_timex[3]

        result_list.append(new_dict)


pd.DataFrame(result_list).to_csv("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_results/K-fold-results_CBERT1.csv")
