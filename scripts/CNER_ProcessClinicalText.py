from CNER_EntityExraction import *
from CNER_BertUtility import *

import pandas as pd

def process_text(text):
    word_list_df = pd.DataFrame(process_string_finetune(text,1))

    entity_extraction = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/event.pkl","event_flag",bert_layer=12)
    entity_extraction_timex = EntityExtraction("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/timex.pkl","timex_flag",downsample_multiplier=4,bert_layer=12)

    print(entity_extraction.predict(word_list_df))
    print(entity_extraction_timex.predict(word_list_df))
    #word_list_df['event_probab'] = entity_extraction.predict_proba(word_list_df)
    #word_list_df['timex_probab'] = entity_extraction.predict_proba(word_list_df)

    #word_list_df['entity_flag'] = word_list_df.apply(lambda x:0 if(x.event_probab<0.5 and x.timex_probab<0.5) else 1 if(x.event_probab>x.timexProbab) else 2, axis=1)

    #word_list_df = word_list_df[word_list_df['entity_flag']!=0]

    #list_of_positions = resolve_continuity(text,word_list_df)