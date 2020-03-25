# DS 5500 Project Proposal: Clinical event and Temporal Information extraction using Contextual word embeddings

Authors: Manas Ranjan Mohanty mohanty.m@husky.neu.edu, Parth Barhanpurkar barhanpurkar.p@husky.neu.edu, Anushka Tak tak.a@husky.neu.edu

Proposed as a Quantiphi project.

# Summary

Understanding the clinical timeline is crucial in determining a patient's diagnosis and treatment. Narrative provider notes from electronic health records frequently detail important information on the temporal ordering of events in a patient's clinical timeline. Temporal analysis of the clinical narrative is therefore a prime target for developing automated natural language processing techniques that allow computerized systems to access, reason about, and sequence the clinical events in a patient's record. Such techniques potentially enable or facilitate tracking disease status, monitoring treatment outcomes and complications, discovering medication side effects, etc. With this intent, The Sixth Informatics for Integrating Biology and the Bedside (i2b2) Natural Language Processing Challenge for Clinical Records focused on the temporal relations in clinical narratives. The organizers provided the research community with a corpus of discharge summaries annotated with temporal information, to be used for the development and evaluation of temporal reasoning systems.

The primary approach taken by all the researchers on this task was to implement a combination of rule based and machine learning based NLP system. The challenge faced while implementing a machine learning based method here is limitations in availability of data and thus the trained model did not generalize well. In this project, for the machine learning part, we propose the usage of one of the pre-trained models(BERT variants such as BioBERT, ClinicalBERT), which are previously trained on a big corpus of biomedical and clinical texts, to generate the word embeddings and perform entity extraction by either fine tuning the upper layers or training a new model by passing the word embeddings to a new network(transfer learning).

# Concepts  

### What are word embeddings?

A word embedding is a learned representation for text where words that have the same meaning have a similar representation.(src- Machinelearningmastery.com). Key to the approach is the idea of using a dense distributed representation for each word.
Each word is represented by a real-valued vector, often tens or hundreds of dimensions. This is contrasted to the thousands or millions of dimensions required for sparse word representations, such as a one-hot encoding.

### Why word embeddings perform poorly on NLP tasks?

Word embdding representations fail to capture the context;i.e. embedding for a word remains the same irresepective of it's sorrounding words. For many NLP tasks, such as Named Entity Recognition, Sentiment Analysis, Question Answering etc the context of the text is the most critical aspect to caputre for building an effective system. Thus, over the years researchers have put thier focus on building models to generate contexutal word embeddings. Machine learning models with complex Neural Network architecture(Such as RNN, LSTM etc) have been trained for this purpose.

### Why it is difficult to build an effective model for NLP from scratch?

Due to the size of vocabulary of languages, it is challenging to train a machine learning language model which generalises well unless the dataset is large enough. Even though the dataset is very large, the challenge with infrastructure requirements for training a model is difficult to tackle with. So it is preferable if a pretrained model is readily available, which can be used with transfer learning/fine tuning for different NLP tasks.

### What is BERT and how BERT works?

BERT(Bidirectional Encoder Representations from Transformers) is one of the state of the art pretrained models available. It is pretrained on Wikipedia corpus, with hidden words and next sentence prediction activities. It's bidirection property makes it better than previously available architectures(RNN, LSTM etc) in terms of capturing the context. BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task.

<p align="center"><img src="https://miro.medium.com/max/1095/0*ViwaI3Vvbnd-CJSQ.png" height="300" width="400"></p>

                                            BERT Training(Predicting the hidden words)
                                            Image source(https://towardsdatascience.com/)

### Pretrained BERT on Clinical Text -

Researchers in Healthcare have taken inspiration from BERT and pretrained models on clinical text and healthcare NLP tasks. BIO-BERT(trained on WIKIPedia, Pubmed articles and MIMIC notes), NCBI-BERT(trained only on PUBMED article and MIMIC notes), Clinical BERT(BIO-BERT finetuned on clinical notes) are examples of such models which are open-sourced. For the purpose of our research, we found NCBI-BERT and Clinical BERT more relevant and thus, we are planning to use these.  


# Data

Data for this problem comes from 2012 Integrating Biology and the Bedside (i2b2) Natural Language Processing Challenge. It contains clinical discharge notes tagged with EVENT(clinical event), TIMEX(Temporal Information), SECTIME(Also Temporal Information), and TLINK(Linking EVENT and TIMEX) tags stored in XML files. For the purpose of this challenge, we treat TIMEX and SECTIME as a single tag representing temporal information. Below is a sample discharge note information presented visually(with out TLINK).


<p align="center"><img src="https://drive.google.com/uc?export=view&id=1Qw8WJqrlXPgiGSpKj32OQSp9eFXP6nGj" height="300" width="500"></p>


                                                  A sample tagged discharge note
                                                        
# Proposed Plan of Research

We propose to complete the project in two phases.

## Stage One

First, we focus on literature survey where we delve deeper into understanding the problem statement, the possible applications it can have to facilitate living, what are the potential roadblocks and challenges we might have to encounter during the project, what are the existing approaches and how we suggest a  novel approach to the same problem. After a thorough research, we will extract all the clinical entities of interest. There are two entities primarily involved in this project; events and temporal expressions. Event here means all clinically relevant events and situations, including symptoms, tests, procedures, and other occurrences. Temporal expressions include all expressions related to time, such as dates, times, frequencies, and durations. To enlist them,

• Define scope for rule based and machine learning based approach

• Generate embeddings for words in a sentence by using at least a sentence before(If available) and a sentence after(If available), along with the sentence itself.

• Train binary classification models to extract all the clinical entities of interest using embeddings from different layers of BERT

• Compare performance from different BERT variants(NCBI and Clinical BERT)

• Identify which attention layer output embedding to use

• Finetune the model for achieving better performance


## Stage Two

For extracting relations between entities extracted from phase 1, we propose to use the ideas suggested by Liang
Yao et al. in the paper KG-BERT(BERT for Knowledge Graph Completion). Knowledge graphs is a knowledge base used by Google and its
services to enhance its search engine's results with information gathered from a variety of sources. At a high level, It is a set of nodes and edges(may be directed or undirected) connecting the nodes, where a node represents an individual data entity and an edge between two nodes represents connection between the entities represented by the node.

KG-BERT proposals: The authors proposed to build a knowledge graph using BERT based classification models. These models can be trained in two possible ways -

a) Training BERT by passing the text [CLS]<entity1>[SEP]<relation name>[SEP]<entity2>[SEP] as an input and label(0 or 1), 1 if there is a relation between the entities specified by relation name and 0 otherwise. The model is then trained to minimise cross entropy loss by
applying logistic regression to the final layer embedding. Here [CLS] and [SEP] are special tokens, identified by BERT as classification
and separator tokens. For a given entry, BERT only uses the embedding for[CLS] token in the last layer to classify the entry.

b) Given there can be n possible types of relation between entities, training BERT by passing the text [CLS]<entity1>[SEP]<entity2>[SEP] as an input and label(0...n), where label represents relationship type. The model is then trained to minimise cross entropy loss of multiclass classification. Similar to above, BERT uses the embedding for[CLS] token in the last layer to classify the entry.


Using embeddings from phase 1: We propose to implement training method b from KG-BERT with a slight variation. Instead of training a BERT model for relation extraction, our method can use the embeddings for the entities generated from phase 1. Then we can train a multi-class classification model by passing <Entity1 Embedding><Entity2 Embedding>. We do not need to use the special tokens here, as the individual embeddings are of fixed size(768 in our case). We are planning to add negative cases for each relationship type by sampling from events and temporal information which do not have any relation between them.

At the end of phase 2, we are planning to form a containerised solution by putting our code along with its dependencies into a docker, which then can be deployed into any Amazon Web Services instance. We suggest Docker as a deployment solution for continuous integration over multiple platforms without any compatibility concerns.It will enable a smooth cycle between development, test, production, and customer environments.


# Evaluation Metrics

## F1 score
F1 score, by definition, serves as a fairly good estimate in the field of information retrieval for document classification. Since our project finds application in text processing, we propose to use F1- score as our evaluation metric.
F1 score is traditionally, the harmonic mean of Precision and Recall. 

As we are planning to solve the problem using binary classification for each entity and the dataset is imbalanced(few words from the text are entities), binary F1 score(considering only the positive labels) gives a true picture of model performance.

For the purpose of this research, we will also track Precision and Recall along with F1 score. In this context, we are planning to aim for higher F1 score, but in case we need to trade-off between better precision or better recall, we will aim for better better precision (less false positives), as false positives might introduce more noise in stage 2.

# Preliminary Results

Number of documents = 310
words = 179983
events = 63757
time entries = 8446

Using 248 documents, We trained a binary classification model using logistic regression, with the output layer(layer-12) word embeddings.

Below are the results of the model.

### Event model 

F1 Score: 0.810
Precision: 0.814
Recall: 0.806

Confusion matrix - 

[[19840  2257] 

 [ 2370  9879]]

### Timex model 

F1 Score: 0.707
Precision: 0.838
Recall: 0.611

Confusion matrix - 

[[32655   178] 

 [  589   924]]
 
# References
 
• 2012 i2b2 challenge: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3756273/

• BERT: https://arxiv.org/abs/1810.04805

• Clinical BERT: https://arxiv.org/abs/1904.03323

• BIOBERT: https://arxiv.org/abs/1901.08746

• KGBERT: https://arxiv.org/abs/1909.03193

• https://en.wikipedia.org/wiki/Natural_language_processing

• https://expertsystem.com/entity-extraction-work/

• https://yashuseth.blog/2019/10/08/introduction-question-answering-knowledge-graphs-kgqa/

• https://www.semanticscholar.org/paper/Named-Entity-Recognition-using-Word-Embedding-as-a-Seok-Song/e4625b1616be1b05fa0fe3427ca4e6d3a8ba9b74

• https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
