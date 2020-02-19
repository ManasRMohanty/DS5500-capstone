from CNER_EntityExraction import *
from CNER_BertUtility import *

import pandas as pd

def process_text(text):
    word_list_df = pd.DataFrame(process_string_finetune(text,1))

    entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_event.pkl","event_flag",bert_layer=12)
    entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/final_timex.pkl","timex_flag",downsample_multiplier=4,bert_layer=12)

    print(entity_extraction.predict(word_list_df))
    print(entity_extraction_timex.predict(word_list_df))

    word_list_df['event_probab'] = [a[1] for a in entity_extraction.predict_proba(word_list_df)]
    word_list_df['timex_probab'] = [a[1] for a in entity_extraction_timex.predict_proba(word_list_df)]

    word_list_df['entity_flag'] = word_list_df.apply(lambda x:0 if(x.event_probab<0.5 and x.timex_probab<0.5) else 1 if(x.event_probab>x.timex_probab) else 2, axis=1)

    word_list_df = word_list_df[word_list_df['entity_flag']!=0]

    print(len(word_list_df))

    list_of_positions = resolve_continuity(text,word_list_df)

    last_position = 0
    processed_text = ""

    for entry in list_of_positions:
        processed_text = processed_text + text[last_position:entry["begin_pos"]]
        
        if(entry["entity_flag"]==1):
            processed_text = processed_text + "<span style=\"background-color:#00FF00;\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
        else:
            processed_text = processed_text + "<span style=\"background-color:cyan;\">"+ text[entry["begin_pos"]:entry["end_pos"]] + "</span>"
        
        last_position = entry["end_pos"]
    

    processed_text = processed_text + text[last_position:]

    processed_text.replace('\n', '<br>').replace('\r', '<br>')
    print("<br />".join(processed_text.split("\n")))
    return "<br />".join(processed_text.split("\n"))