from CNER_BertUtility import *

def is_available_in_pos_list(dict_list, begin_pos, end_pos):
    for entry in dict_list:
        if(entry["start"]<=begin_pos and entry["end"]>=end_pos):
            return True
    
    return False


class DischargeNote():
    def __init__(self, root):
        self.xml_root = root
           
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