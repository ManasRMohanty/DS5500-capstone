import nltk
import numpy as np

from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch

'''
This class contains all the utility methods for the project.

get_bert_token_positions(input_text,token_list,start_from_pos=0) - 
This method accepts an input text and a list of tokens, then finds positions of all the tokens which 
completes the input text. For example, given input text 'xyz' and a list of tokens ['ab','cd','ef,'g','xy','z'], 
this method matches 'xyz' with 'xy' and 'z', and thus returns [4,5], which are the positions corresponding to tokens 'xy' 
and 'z' in the list. start_from_pos is an optional parameter to indicate the starting position in the list to start the 
search for.

process_string(string_input, padding_length) - 

This method accepts a string input as first parameter. At first it devides the string input into sentences. Then, for a given  
sentence it considers few sentences before and after it. The number of sentences to consider is defined by padding_length 
argument. If a sentence is the first sentence of the sequence, it only considers sentences to its right, similarly if a sentence 
is the last sentence of the sequence it considers only to its left. After that, the set of sentences are passed to BERT to 
generate bert token embeddings. Then for the given sentence(in the middle), the word embeddings for all the words are calculated one
by one, by matching the corresponding BERT tokens, and then averaging the embeddings associated with the bert tokens. This process is
repeated for all the sentences and word embeddings are generated.

'''
def get_bert_token_positions(input_text,token_list,start_from_pos=0):
    
    pos_list = []                    
    
    name_to_match = input_text.lower().replace(" ","")
    remaining_name = input_text.lower().replace(" ","")
    
    name = ""
    count = start_from_pos

    for entry in token_list[start_from_pos:]:
        if(remaining_name.startswith(entry.strip("##").lower())):
            pos_list.append(count)
            remaining_name = remaining_name[len(entry.strip("##").lower()):]
            name = name + entry.strip("##").lower()
            if(name == name_to_match):
                break
        else:
            pos_list = []
            name = ""
            remaining_name = name_to_match
            if(remaining_name.startswith(entry.strip("##").lower())):                                 
                pos_list.append(count)                                                                
                remaining_name = remaining_name[len(entry.strip("##").lower()):]
                name = name + entry.strip("##").lower()    
                if(name == name_to_match):                                   
                    break

        count = count + 1

    return pos_list

def process_string(string_input, padding_length):

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(string_input)
    
    config = BertConfig.from_pretrained('bio-bert-uncased')
    config.output_hidden_states = True

    bert_tokenizer = BertTokenizer.from_pretrained('bio-bert-uncased')
    bert_model = BertModel.from_pretrained('bio-bert-uncased',config=config)
    
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
