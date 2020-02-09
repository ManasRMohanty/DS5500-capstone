from CNER_EntityExraction import EntityExtraction

import os
import xml.etree.ElementTree as ET
import pandas as pd

word_list = []
count = 0
for file in os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/train_data/"):
    if file.endswith(".xml"):
        try:
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/train_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = Discharge_note(root)
            discharge_note.process_note()
            word_list.extend(discharge_note.processed_text)
            count = count + 1
            print(count)

        except:
            count = count + 1
            print("Error in processing file:",file)


word_list_test = []
count = 0
for file in os.listdir("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/"):
    if file.endswith(".xml"):
        try:
            file_name = os.path.join("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/test_data/", file)
            tree = ET.parse(file_name)
            root = tree.getroot()
            discharge_note = Discharge_note(root)
            discharge_note.process_note()
            word_list_test.extend(discharge_note.processed_text)
            count = count + 1
            print(count)
            
        except:
            count = count + 1
            print("Error in processing file:",file)


for bert_layer in range(13):
    print("Layer:",bert_layer)

    entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/event.pkl","event_flag",bert_layer=bert_layer)
    entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/timex.pkl","timex_flag",downsample_multiplier=4,bert_layer=bert_layer)
    
    entity_extraction.fit(pd.DataFrame(word_list))
    entity_extraction_timex.fit(pd.DataFrame(word_list))
    
    entity_extraction.test_data(pd.DataFrame(word_list_test))
    entity_extraction_timex.test_data(pd.DataFrame(word_list_test))
