import nltk
import numpy as np
import re

from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch
import pickle
from keras.preprocessing.sequence import pad_sequences

bert_model = pickle.load(open("C:/Users/itsma/Documents/Capstone project/DS5500-capstone/models/entity_model_finetuned_complete.pkl","rb"))
bert_model.cpu()

'''
This class contains all the utility methods for the project.

get_bert_token_positions(input_text,token_list,start_from_pos=0,prior_partial_word="") - 
This method accepts an input text and a list of tokens, then finds positions of all the tokens which 
completes the input text. For example, given input text 'xyz' and a list of tokens ['ab','cd','ef,'g','xy','z'], 
this method matches 'xyz' with 'xy' and 'z', and thus returns [4,5], which are the positions corresponding to tokens 'xy' 
and 'z' in the list. start_from_pos is an optional parameter to indicate the starting position in the list to start the 
search for. prior_partial_word is a parameter that is used in case more than one word is covered by a single bert token, such as 
'cannot' is a token that covers 'can' and 'not'.

process_string_finetune(string_input, padding_length) - 

This method accepts a string input as first parameter. At first it devides the string input into sentences. Then, for a given  
sentence it considers few sentences before and after it. The number of sentences to consider is defined by padding_length 
argument. If a sentence is the first sentence of the sequence, it only considers sentences to its right, similarly if a sentence 
is the last sentence of the sequence it considers only to its left. After that, the set of sentences are passed to BERT to 
generate bert token embeddings. Then for the given sentence(in the middle), the word embeddings for all the words are calculated one
by one, by matching the corresponding BERT tokens, and then averaging the embeddings associated with the bert tokens. This process is
repeated for all the sentences and word embeddings are generated.

'''
def get_bert_token_positions(input_text,token_list,start_from_pos=0,prior_partial_word=""):
    partial_word = ""

    pos_list = []                    
    
    if(prior_partial_word!=""):
        input_text = prior_partial_word + input_text 

    name_to_match = input_text.lower().replace(" ","")
    remaining_name = input_text.lower().replace(" ","")
    
    name = ""
    count = start_from_pos

    for entry in token_list[start_from_pos:]:
        entry_text = entry.strip("##").lower()
        if(entry_text.startswith(remaining_name) and (entry_text != remaining_name)):
            partial_word = remaining_name
            pos_list.append(count)
            break
             
        if(remaining_name.startswith(entry_text)):
            pos_list.append(count)
            remaining_name = remaining_name[len(entry_text):]
            name = name + entry_text
            if(name == name_to_match):
                break
        else:
            pos_list = []
            name = ""
            remaining_name = name_to_match
            if(remaining_name.startswith(entry_text)):                                 
                pos_list.append(count)                                                                
                remaining_name = remaining_name[len(entry_text):]
                name = name + entry_text   
                if(name == name_to_match):                                   
                    break

        count = count + 1
    
    return [pos_list,partial_word]


def process_string(string_input, padding_length):


    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(string_input)
    
    config = BertConfig.from_pretrained('C:/Users/itsma/Documents/BERT_models/Clinical_BERT')
    config.output_hidden_states = True

    bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/itsma/Documents/BERT_models/Clinical_BERT')
    bert_model = BertModel.from_pretrained('C:/Users/itsma/Documents/BERT_models/Clinical_BERT',config=config)
    
    positions_covered = 0
    word_list = []
    
    for index in range(len(sentences)):
        new_dict_sentence = {}
        sentence = sentences[index]
        new_dict_sentence["sentence"] = sentence
        new_dict_sentence["padding_length"] = padding_length
        start_index_bert = max(0,index-padding_length)
        end_index_bert = min(len(sentences),index+padding_length)

        bert_input = ' '.join(sentences[start_index_bert:(end_index_bert+1)])

        encodings = bert_tokenizer.encode(bert_input,add_special_tokens = True)

        if(len(encodings)>=512):
            encodings = encodings[0:512]
        input_ids = torch.tensor(encodings).unsqueeze(0)  
        outputs = bert_model(input_ids)
        bert_vector = outputs[2]
        bert_tokens = bert_tokenizer.convert_ids_to_tokens(encodings) #bert_tokenizer.tokenize(bert_input,add_special_tokens = True)

        start_pos = 0
        prior_pos = get_bert_token_positions(' '.join(sentences[start_index_bert:index]),bert_tokens)
        
        if(len(prior_pos)>0):
            start_pos = max(prior_pos)
            
        tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.pos_tag(tokens)

        sentence_covered = ''
        
        
        for token in pos_tokens:
            new_dict = {}
            current_word = token[0]
            new_dict["word"] = current_word
            token_position = string_input.find(current_word, positions_covered)
            spaces_between = string_input[positions_covered:token_position] 
            sentence_covered = sentence_covered + spaces_between + current_word
            positions_covered = token_position + len(current_word)
            new_dict["begin_pos"] = token_position
            new_dict["end_pos"] = positions_covered

            bert_token_positions = get_bert_token_positions(current_word,bert_tokens,start_pos)
            
            vec_list_layers = []
            
            if(len(bert_token_positions)==0):
                continue

            start_pos = bert_token_positions[-1] + 1
            
            for bert_layer in range(13):
                vec_list = []
                for entry in bert_token_positions:
                    vec_list.append(bert_vector[bert_layer][0][entry].data.numpy())
                
                vec_list_layers.append(np.mean(vec_list,axis=0))
                
            new_dict["keyword_vector"] = vec_list_layers

            word_list.append(new_dict)
    
    return word_list

def process_string_finetune(string_input, padding_length,output_layer_only = False):
    string_input = re.sub(r'Admission Date :\n([0-9/ ]*)\n', 'Admission Date : \g<1>\n', string_input)
    string_input = re.sub(r'Discharge Date :\n([0-9/ ]*)\n', 'Discharge Date : \g<1>\n', string_input)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    init_sentences = tokenizer.tokenize(string_input)

    sentences = []
    
    for entry in init_sentences:
        sentences.extend(entry.split("\n"))
    
    #config = BertConfig.from_pretrained('C:/Users/itsma/Downloads/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000')
    #config.output_hidden_states = True

    bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/itsma/Documents/BERT_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12')
    #bert_model = BertModel.from_pretrained('C:/Users/itsma/Downloads/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000',config=config)

    positions_covered = 0
    word_list = []
    
    for index in range(len(sentences)):
        sentence = sentences[index]
        start_index_bert = max(0,index-padding_length)
        end_index_bert = min(len(sentences),index+padding_length)

        bert_input = ' '.join(sentences[start_index_bert:(end_index_bert+1)])

        encodings = bert_tokenizer.encode(bert_input,add_special_tokens = True)
        
        if(len(encodings)>=474):
            encodings = encodings[0:474]

        input_ids = torch.tensor(encodings).long().unsqueeze(0)
        
        outputs = bert_model(input_ids,token_type_ids=None)
    
        bert_vector = outputs[1]

        bert_tokens = bert_tokenizer.convert_ids_to_tokens(encodings) #bert_tokenizer.tokenize(bert_input,add_special_tokens = True)

        start_pos = 0
        prior_pos = get_bert_token_positions(' '.join(sentences[start_index_bert:index]),bert_tokens)[0]
        
        if(len(prior_pos)>0):
            start_pos = max(prior_pos)
            
        tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.pos_tag(tokens)

        sentence_covered = ''
        prior_partial_word = ''
        for token in pos_tokens:
            new_dict = {}
            current_word = token[0]
            new_dict["word"] = current_word
            
            [bert_token_positions, partial_word] = get_bert_token_positions(current_word,bert_tokens,start_pos,prior_partial_word)
            
            vec_list_layers = []
            
            if(len(bert_token_positions)==0):
                prior_partial_word = ""
                continue
            if(partial_word):
                prior_partial_word = partial_word
                start_pos = bert_token_positions[-1]
            else:
                prior_partial_word = ""
                start_pos = bert_token_positions[-1] + 1
            
            token_position = string_input.find(current_word, positions_covered)
            spaces_between = string_input[positions_covered:token_position] 
            sentence_covered = sentence_covered + spaces_between + current_word
            positions_covered = token_position + len(current_word)
            new_dict["begin_pos"] = token_position
            new_dict["end_pos"] = positions_covered
            new_dict["sentence_index"] = index
            
            if(output_layer_only):
                vec_list = []
                for entry in bert_token_positions:
                    vec_list.append(bert_vector[12][0][entry].data.numpy())
                
                vec_word = np.mean(vec_list,axis=0)
                new_dict["keyword_vector"] = vec_word
            else:
                for bert_layer in range(13):
                    vec_list = []
                    for entry in bert_token_positions:
                        vec_list.append(bert_vector[bert_layer][0][entry].data.numpy())
                    
                    vec_list_layers.append(np.mean(vec_list,axis=0))
                
                new_dict["keyword_vector"] = vec_list_layers
            
            word_list.append(new_dict)
    
    return word_list, sentences