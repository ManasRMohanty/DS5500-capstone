import nltk
import numpy as np

from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch

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
    
    config = BertConfig.from_pretrained('C:/Users/itsma/Documents/BERT_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12')
    config.output_hidden_states = True

    bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/itsma/Documents/BERT_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12')
    bert_model = BertModel.from_pretrained("C:/Users/itsma/Documents/BERT_models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",config=config)
    
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

        input_ids = torch.tensor(bert_tokenizer.encode(bert_input)).unsqueeze(0)  
        outputs = bert_model(input_ids)
        bert_vector = outputs[2][0][0].data.numpy()
        bert_tokens = bert_tokenizer.tokenize(bert_input)

        start_pos = 0
        prior_pos = get_token_positions(' '.join(sentences[start_index_bert:index]),bert_tokens)
        
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

            bert_token_positions = get_token_positions(current_word,bert_tokens,start_pos)
            vec_list = []
            if(len(bert_token_positions)==0):
                continue
            start_pos = bert_token_positions[-1] + 1

            for entry in bert_token_positions:
                vec_list.append(bert_vector[entry])
            
            bert_vector_word = np.mean(vec_list,axis=0)
            new_dict["keyword_vector"] = bert_vector_word

            word_list.append(new_dict)
    
    return word_list