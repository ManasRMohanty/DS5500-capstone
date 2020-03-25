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
print("File reading for training in progress...")

word_list_train= []
count = 0
pbar = ProgressBar()
result_list = []

for file in pbar(os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/train_data/")):
    if file.endswith(".xml"):
        try:
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/train_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = DischargeNote(root,file,baseline=False)
            discharge_note.process_note()
            word_list_train.extend(discharge_note.processed_text)
        except:
            print("Error processing file:",file)

word_list_train_df = pd.DataFrame(word_list_train)

print("File reading for testing in progress...")

word_list_test= []
count = 0
pbar = ProgressBar()

for file in pbar(os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/")):
    if file.endswith(".xml"):
        try:
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = DischargeNote(root,file,baseline=False)
            discharge_note.process_note()
            word_list_test.extend(discharge_note.processed_text)
        except:
            print("Error processing file:",file)

word_list_test_df = pd.DataFrame(word_list_test)


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


new_dict = {}
new_dict['bert_layer'] = 12

print("Layer:",12)

entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/event.pkl","event_flag",last_layer_only=True)
entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/timex.pkl","timex_flag",last_layer_only=True)

entity_extraction.fit(word_list_train_df)
entity_extraction_timex.fit(word_list_train_df)

print("Training data results")
entity_extraction.test_data(word_list_train_df)
entity_extraction_timex.test_data(word_list_train_df)

print("Test data results")
test_results_event = entity_extraction.test_data(word_list_test_df)
test_results_timex = entity_extraction_timex.test_data(word_list_test_df)

'''
new_dict['event_model_f1'] = test_results_event[1]
new_dict['event_model_precision'] = test_results_event[2]
new_dict['event_model_recall'] = test_results_event[3]

new_dict['timex_model_f1'] = test_results_timex[1]
new_dict['timex_model_precision'] = test_results_timex[2]
new_dict['timex_model_recall'] = test_results_timex[3]

result_list.append(new_dict)

#pd.DataFrame(result_list).to_csv("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_results/K-fold-results_fine_tune_comb.csv")
'''