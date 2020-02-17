# DS 5500 Project Proposal: Clinical event and Temporal Information extraction using Contextual word embeddings

Authors: Manas Ranjan Mohanty mohanty.m@husky.neu.edu, Parth Barhanpurkar barhanpurkar.p@husky.neu.edu, Anushka Tak tak.a@husky.neu.edu

Proposed as a Quantiphi project.

# Summary

Understanding the clinical timeline is crucial in determining a patient's diagnosis and treatment. Narrative provider notes from electronic health records frequently detail important information on the temporal ordering of events in a patient's clinical timeline. Temporal analysis of the clinical narrative is therefore a prime target for developing automated natural language processing techniques that allow computerized systems to access, reason about, and sequence the clinical events in a patient's record. Such techniques potentially enable or facilitate tracking disease status, monitoring treatment outcomes and complications, discovering medication side effects, etc. With this intent, The Sixth Informatics for Integrating Biology and the Bedside (i2b2) Natural Language Processing Challenge for Clinical Records focused on the temporal relations in clinical narratives. The organizers provided the research community with a corpus of discharge summaries annotated with temporal information, to be used for the development and evaluation of temporal reasoning systems.

The primary approach taken by all the researchers on this task was to implement a combination of rule based and machine learning based NLP system. The challenge faced while implementing a machine learning based method here is limitations in availability of data and thus the trained model did not generalize well. In this project, for the machine learning part, we propose the usage of one of the pre-trained models(BERT variants such as BioBERT, ClinicalBERT), which are previously trained on a big corpus of biomedical and clinical texts, to generate the word embeddings and perform entity extraction by either fine tuning the upper layers or training a new model by passing the word embeddings to a new network(transfer learning).

# Concepts  
connection missing 

##What are word embeddings?

A word embedding is a learned representation for text where words that have the same meaning have a similar representation.(src- Machinelearningmastery.com). Key to the approach is the idea of using a dense distributed representation for each word.
Each word is represented by a real-valued vector, often tens or hundreds of dimensions. This is contrasted to the thousands or millions of dimensions required for sparse word representations, such as a one-hot encoding.

Why word embeddings perform poorly on NLP tasks?

Word embdding representations fail to capture the context;i.e. embedding for a word remains the same irresepective of it's sorrounding words. For many NLP tasks, such as Named Entity Recognition, Sentiment Analysis, Question Answering etc the context of the text is the most critical aspect to caputre for building an effective system. Thus, over the years researchers have put thier focus on building models to generate contexutal word embeddings. Machine learning models with complex Neural Network architecture(Such as RNN, LSTM etc) have been trained for this purpose.

Why it is difficult to build an effective model for NLP from scratch?

Due to the size of vocabulary of languages, it is challenging to train a machine learning language model which generalises well unless the dataset is large enough. Even though the dataset is very large, the challenge with infrastructure requirements for training a model is difficult to tackle with. So it is preferable if a pretrained model is readily available, which can be used with transfer learning/fine tuning for different NLP tasks.

What is BERT and how BERT works?

BERT(Bidirectional Encoder Representations from Transformers) is one of the state of the art pretrained models available. It is pretrained on Wikipedia corpus, with hidden words and next sentence prediction activities. It's bidirection property makes it better than previously available architectures(RNN, LSTM etc) in terms of capturing the context. BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task.

Include image from ppt
![BERT Architecture](https://miro.medium.com/max/1095/0*ViwaI3Vvbnd-CJSQ.png)

# Proposed Plan of Research

We propose to complete the project in two phases.


## Stage One

First, we focus on literature survery where we delve deeper into understanding the problem statement, the possible applications it can have to facilitate living, what are the potential roadblocks and challenges we might have to encounter during the project, what are the existing approaches and how we suggest a  novel approach to the same problem. After a thorough reasearch, we will extract all the clinical entities of interest. There are two entities primarily involved in this project; events and temporal expressions. Event here means all clinically relevant events and situations, including symptoms, tests, procedures, and other occurrences. Temporal expressions include all expressions related to time, such as dates, times, frequencies, and durations. To enlist them,

• Define scope for rule based and machine learning based approach

• Extract all the clinical entities of interest

• Identify which attention layer output embedding to use

• Compare performance from different clinical BERT variant based approach

## Stage Two

In the second stage, we plan to develop machine learning models to extract temporal relations (e.g. before, after, simultaneous, etc.) that hold between different events or between events and temporal expressions. In case time permits, we are planning to use the word embeddings to build a knowledge graph in order to experiment and check the benefits. All our models will be hosted using AWS/GCP. We alos aim to make a Web based application to run this model to extract temporal relations deployed on Flask.

# Evaluation Metrics

## F1 score
F1 score, by definition, serves as a fairly good estimate in the field of information retrieval for document classification. Since our project finds application in text processing, we propose to use F1- score as our evaluation metric.
F1 score is traditionally, the harmonic mean of Precision and Recall. 

more here. about how this is useful for us here. 

test data entitites capture??

# Preliminary Results

#docs?
#words
#events
#time entries
#baseline model ran on the outer layer
layer 12 output
retrain on next layers

![Sample Discharge Note](/DS5500-capstone/Sample Discharge note.JPG)

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
