from CNER_BertUtility import *
import numpy as np


'''
This class is for initializing and processing the discharge notes. A discharge note is initialized using the XML
associated with it. The two main tags of concern are 'TEXT' and 'TAGS'. The 'TEXT' tag contains the actual text linked
to the discharge note. It has admission, discharge dates along with history of present illness information. This part is
processed to generate word embeddings. 'TAGS' section contains information about, four different tas; 'EVENT', 'TIMEX3',
'SECTIME' and 'TLINK'. 'EVENT' corresponds to mention of a clinical event in the given text. 'TIMEX' and 'SECTIME' present
any time related mention in the givent text. All these above tags contains the begin and end positions of the tag in text.
Also each mention of these tags has its own id.

'TLINK' tag, links events to its corresponding time information. For this purpose, it uses the id of the corresponding tags.

The class is to read the XML and gather all these information. Below are the details about methods in this file.

is_available_in_pos_list -
This method accepts a list of tags, a begining and an ending position as arguments. Then, it iterates through the list to find
out whether any tag falls under the given begining and ending position.

get_entity_from_id - 
This method accepts an ID and returns an event or timex entity corresponsing to that ID if available.

get_relation_from_ids - 
This method accepts a from and to id, and returns a tlink if available between the entities corresponding to the IDs.

process_note -
This method processes the XML file. At first it generates word embeddings for all the words in the text(available under 'TEXT' tag).
Then it indentifies all the words which are tagged as events or time. Note - Both 'TIMEX' and 'SECTIME' tags are considered as time
tags.

After the embeddings for the words are generated, through the tlink information available, this method generates all the relations, which
would be later used to train model for relation extraction. Vectors to represent a relation are concatenation of from and to entity vectors.
Relationship type is used as a class for multiclass classification. 

For understanding how the file looks like, we have attached a sample xml in git, under sample file.
'''


def is_available_in_pos_list(dict_list, begin_pos, end_pos):
    id_list = []
    is_available = False
    for entry in dict_list:
        if(entry["start"]<=begin_pos and entry["end"]>=end_pos):
            id_list.append(entry["id"])
            is_available = True

    return [is_available,id_list]

class DischargeNote():
    def __init__(self, root,file_name,baseline=True):
        self.xml_root = root
        self.file_name = file_name
        self.baseline = baseline

    def get_entity_from_id(self,id):
        if(id.startswith("E")):
            for event in self.event_list:
                if(event['id']==id):
                    return event
        else:
            for timex in self.timex_list:
                if(timex['id']==id):
                    return timex

    def get_relation_from_ids(self,id1,id2):
        for tlink in self.tlink_list:
            if(tlink['fromID']==id1 and tlink['toID']==id2):
                return ['Direct',tlink['type'],tlink['id']]
            elif(tlink['fromID']==id2 and tlink['toID']==id1):
                return ['Inverse',tlink['type'],tlink['id']]
            
        return ['NA', 'No_Rel','NA']

    def process_note(self):
        root = self.xml_root
        text_section = root.find('TEXT')
        text = text_section.text
        
        self.text = text

        if(self.baseline):
            self.processed_text, self.sentences = process_string(text,1)
        else:
            self.processed_text, self.sentences = process_string_finetune(text,1,output_layer_only=True)
            
        tag_section = root.find('TAGS')
        event_list = []
        timex_list = []
        tlink_list = []

        for child in tag_section:
            if(child.tag=='EVENT'):
                event_list.append(child.attrib)
            elif(child.tag=='TIMEX3'):
                timex_list.append(child.attrib)
            elif(child.tag=='TLINK'):
                tlink_list.append(child.attrib)
        """
        They are stored as strings. So we are converting them to integers
        """
        for sub in event_list:
            sub["start"] = int(sub["start"])
            sub["end"] = int(sub["end"])

        for sub in timex_list:
            sub["start"] = int(sub["start"])
            sub["end"] = int(sub["end"])


        self.event_list = sorted(event_list, key = lambda i: i['start'])
        self.timex_list = sorted(timex_list, key = lambda i: i['start'])
        self.tlink_list = tlink_list

        for entry in self.processed_text:
            
            event_flag = 0
            timex_flag = 0
            entity_ids = []

            event_entry_available = is_available_in_pos_list(event_list,entry["begin_pos"],entry["end_pos"])
            timex_entry_available = is_available_in_pos_list(timex_list,entry["begin_pos"],entry["end_pos"])

            if(event_entry_available[0]):
                event_flag = 1
                entity_ids.extend(event_entry_available[1])
            if(timex_entry_available[0]):
                timex_flag = 1
                entity_ids.extend(timex_entry_available[1])
        
            entry.update({"timex_flag":timex_flag})
            entry.update({"event_flag":event_flag})
            entry.update({"entity_ids":entity_ids})

            entry.update({"file_name":self.file_name})

        ids_list = []
        for event in self.event_list:
            ids_list.append(event['id'])

            vec_list = []
            sentence_index = -1
            for entry in self.processed_text:
                if(event['id'] in entry['entity_ids']):
                    vec_list.append(entry["keyword_vector"])
                    sentence_index = entry["sentence_index"]
            event.update({"sentence_index":sentence_index})
            event.update({"keyword_vector":np.mean(vec_list,axis=0)})
        
        for timex in timex_list:
            ids_list.append(event['id'])
            vec_list = []
            sentence_index = -1
            for entry in self.processed_text:
                if(timex['id'] in entry['entity_ids']):
                    vec_list.append(entry["keyword_vector"])
                    sentence_index = entry["sentence_index"]
            timex.update({"sentence_index":sentence_index})
            timex.update({"keyword_vector":np.mean(vec_list,axis=0)})

        relation_list = []
        
        for tlink in self.tlink_list:
            new_dict = {}
            from_entity = self.get_entity_from_id(tlink['fromID'])
            to_entity = self.get_entity_from_id(tlink['toID'])
            try:
                if(len(from_entity["keyword_vector"])!=768 or len(to_entity["keyword_vector"])!=768):
                    print(self.file_name)
                    print(from_entity['id'])
                    print(to_entity['id'])
                    print(len(from_entity["keyword_vector"]))
            except:
                print(self.file_name)
                print(tlink['fromID'])
                print(tlink['toID'])

            new_dict['relation_vector'] = np.concatenate((from_entity["keyword_vector"], to_entity["keyword_vector"]), axis=0)
            new_dict['from_sentence_index'] = from_entity["sentence_index"]
            new_dict['to_sentence_index'] = to_entity["sentence_index"]
            new_dict['relation_type'] = tlink['type']
            new_dict['relation_id'] = ('SECTIME' if tlink['id'].lower().startswith('sectime') else 'TL')
            new_dict['file_name'] = self.file_name
            relation_list.append(new_dict)
        
        self.relation_list = relation_list 
        