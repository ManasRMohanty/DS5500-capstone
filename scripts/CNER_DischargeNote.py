from CNER_BertUtility import *


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

process_note -
This method processes the XML file. At first it generates word embeddings for all the words in the text(available under 'TEXT' tag).
Then it indentifies all the words which are tagged as events or time. Note - Both 'TIMEX' and 'SECTIME' tags are considered as time
tags.

For understanding how the file looks like, we have attached a sample xml in git, under sample file.
'''


def is_available_in_pos_list(dict_list, begin_pos, end_pos):
    for entry in dict_list:
        if(entry["start"]<=begin_pos and entry["end"]>=end_pos):
            return True

    return False


class DischargeNote():
    def __init__(self, root,file_name):
        self.xml_root = root
        self.file_name = file_name
           
    def process_note(self):
        root = self.xml_root
        text_section = root.find('TEXT')
        text = text_section.text

        self.processed_text = process_string(text,1)

        tag_section = root.find('TAGS')
        event_list = []
        timex_list = []
        tlink_list = []
        sectime_list = []
        for child in tag_section:
            if(child.tag=='EVENT'):
                event_list.append(child.attrib)
            elif(child.tag=='TIMEX3'):
                timex_list.append(child.attrib)
            elif(child.tag=='TLINK'):
                tlink_list.append(child.attrib)
            elif(child.tag=='SECTIME'):
                sectime_list.append(child.attrib)
        """
        They are stored as strings. So we are converting them to integers
        """
        for sub in event_list:
            sub["start"] = int(sub["start"])
            sub["end"] = int(sub["end"])

        for sub in timex_list:
            sub["start"] = int(sub["start"])
            sub["end"] = int(sub["end"])

        for sub in sectime_list:
            sub["start"] = int(sub["start"])
            sub["end"] = int(sub["end"])


        event_list = sorted(event_list, key = lambda i: i['start'])
        timex_list = sorted(timex_list, key = lambda i: i['start'])
        sectime_list = sorted(sectime_list, key = lambda i: i['start'])

        for entry in self.processed_text:
            event_entry_available = is_available_in_pos_list(event_list,entry["begin_pos"],entry["end_pos"])
            timex_entry_available = is_available_in_pos_list(timex_list,entry["begin_pos"],entry["end_pos"])
            sectime_entry_available = is_available_in_pos_list(sectime_list,entry["begin_pos"],entry["end_pos"])

            entry.update({"event_flag":1 if event_entry_available else 0})
            entry.update({"timex_flag":1 if (timex_entry_available or sectime_entry_available) else 0})
            entry.update({"file_name":self.file_name})


