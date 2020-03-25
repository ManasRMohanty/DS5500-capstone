from CNER_EntityExraction import *
from CNER_BertUtility import *
from pylab import *
import pandas as pd
from CNER_DischargeNote import DischargeNote

import os
import xml.etree.ElementTree as ET

def process_text_from_xml(text):
    
    word_list_df = pd.DataFrame(process_string_finetune(text,1))

    file_name = "C:/Users/itsma/Documents/Capstone project/DS5500-capstone/all_data/" + text +  ".xml"
    tree = ET.parse(file_name)
    root = tree.getroot()
    discharge_note = DischargeNote(root,text,baseline=False)
    discharge_note.process_note()
    text = discharge_note.text
    word_list_df = pd.DataFrame(discharge_note.processed_text)

    entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_event.pkl","event_flag",last_layer_only=True)
    entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_timex.pkl","timex_flag",last_layer_only=True)

    word_list_df['event_probab'] = [a[1] for a in entity_extraction.predict_proba(word_list_df)]
    word_list_df['timex_probab'] = [a[1] for a in entity_extraction_timex.predict_proba(word_list_df)]

    word_list_df['entity_flag'] = word_list_df.apply(lambda x:0 if(x.event_probab<0.5 and x.timex_probab<0.5) else 1 if(x.event_probab>x.timex_probab) else 2, axis=1)

    word_list_df = word_list_df[word_list_df['entity_flag']!=0]

    list_of_positions = resolve_continuity(text,word_list_df)

    event_cmap = cm.get_cmap('YlOrBr', 1000)
    timex_cmap = cm.get_cmap('BuGn', 1000)

    last_position = 0
    processed_text = ""

    for entry in list_of_positions:
        processed_text = processed_text + text[last_position:entry["begin_pos"]]
        
        if(entry["entity_flag"]==1):
            if(entry["event_probab"]>0.9):
                if(entry["event_flag"]==1):
                    color_code = matplotlib.colors.rgb2hex(event_cmap(int((round(entry["event_probab"],3)*1000)-500))[:3])
                else:
                    color_code = "red"
                processed_text = processed_text + "<span style=\"background-color:"+color_code+";\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
            else:
                processed_text = processed_text +  text[entry["begin_pos"]:entry["end_pos"]] 
        else:
            if(entry["timex_probab"]>0.9):
                if(entry["timex_flag"]==1):
                    color_code = matplotlib.colors.rgb2hex(timex_cmap(int((round(entry["timex_probab"],3)*1000)-500))[:3])
                else:
                    color_code = "cyan"
                processed_text = processed_text + "<span style=\"background-color:"+color_code+";\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
            else:
                processed_text = processed_text +  text[entry["begin_pos"]:entry["end_pos"]] 
        
        last_position = entry["end_pos"]
    

    processed_text = processed_text + text[last_position:]

    processed_text.replace('\n', '<br>').replace('\r', '<br>')
    print("<br />".join(processed_text.split("\n")))
    return "<br />".join(processed_text.split("\n"))

def process_text(text):
    if(len(text)<=3):
        return process_text_from_xml(text)

    word_list, sentences = process_string_finetune(text,1, output_layer_only = True)
    word_list_df = pd.DataFrame(word_list)

    entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_event.pkl","event_flag",last_layer_only=True)
    entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_timex.pkl","timex_flag",last_layer_only=True)

    word_list_df['event_probab'] = [a[1] for a in entity_extraction.predict_proba(word_list_df)]
    word_list_df['timex_probab'] = [a[1] for a in entity_extraction_timex.predict_proba(word_list_df)]

    word_list_df['entity_flag'] = word_list_df.apply(lambda x:0 if(x.event_probab<0.5 and x.timex_probab<0.5) else 1 if(x.event_probab>x.timex_probab) else 2, axis=1)

    word_list_df = word_list_df[word_list_df['entity_flag']!=0]

    list_of_positions = resolve_continuity(text,word_list_df)

    event_cmap = cm.get_cmap('YlOrBr', 1000)
    timex_cmap = cm.get_cmap('BuGn', 1000)

    last_position = 0
    processed_text = ""

    for entry in list_of_positions:
        processed_text = processed_text + text[last_position:entry["begin_pos"]]
        
        if(entry["entity_flag"]==1):
            if(entry["event_probab"]>0.9):
                color_code = matplotlib.colors.rgb2hex(event_cmap(int((round(entry["event_probab"],3)*1000)-500))[:3])
                processed_text = processed_text + "<span style=\"background-color:"+color_code+";\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
            else:
                processed_text = processed_text +  text[entry["begin_pos"]:entry["end_pos"]] 
        else:
            if(entry["timex_probab"]>0.9):
                color_code = matplotlib.colors.rgb2hex(timex_cmap(int((round(entry["timex_probab"],3)*1000)-500))[:3])
                processed_text = processed_text + "<span style=\"background-color:"+color_code+";\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
            else:
                processed_text = processed_text +  text[entry["begin_pos"]:entry["end_pos"]] 
        
        last_position = entry["end_pos"]
    

    processed_text = processed_text + text[last_position:]

    processed_text.replace('\n', '<br>').replace('\r', '<br>')
    
    return "<br />".join(processed_text.split("\n"))
