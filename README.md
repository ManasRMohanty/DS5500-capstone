# DS 5500 Project Proposal: Clinical event and Temporal Information extraction using Contextual word embeddings

Authors: Manas Ranjan Mohanty mohanty.m@husky.neu.edu, Parth Barhanpurkar barhanpurkar.p@husky.neu.edu, Anushka Tak tak.a@husky.neu.edu

Proposed as a Quantiphi project.

# Summary

Understanding the clinical timeline is crucial in determining a patient's diagnosis and treatment. Narrative provider notes from electronic health records frequently detail important information on the temporal ordering of events in a patient's clinical timeline. Temporal analysis of the clinical narrative is therefore a prime target for developing automated natural language processing techniques that allow computerized systems to access, reason about, and sequence the clinical events in a patient's record. Such techniques potentially enable or facilitate tracking disease status, monitoring treatment outcomes and complications, discovering medication side effects, etc. With this intent, The Sixth Informatics for Integrating Biology and the Bedside (i2b2) Natural Language Processing Challenge for Clinical Records focused on the temporal relations in clinical narratives. The organizers provided the research community with a corpus of discharge summaries annotated with temporal information, to be used for the development and evaluation of temporal reasoning systems.

The primary approach taken by all the researchers on this task was to implement a combination of rule based and machine learning based NLP system. The challenge faced while implementing a machine learning based method here is limitations in availability of data and thus the trained model did not generalize well. In this project, for the machine learning part, we propose the usage of one of the pre-trained models(BERT variants such as BioBERT, ClinicalBERT), which are previously trained on a big corpus of biomedical and clinical texts, to generate the word embeddings and perform entity extraction by either fine tuning the upper layers or training a new model by passing the word embeddings to a new network(transfer learning).

Our focus during the first half of the project will be to extract all the clinical entities of interest. There are two entities primarily involved in this project; events and temporal expressions. Event here means all clinically relevant events and situations, including symptoms, tests, procedures, and other occurrences. Temporal expressions include all expressions related to time, such as dates, times, frequencies, and durations.

In the second phase of the project, we are planning to develop machine learning models to extract temporal relations (e.g. before, after, simultaneous, etc.) that hold between different events or between events and temporal expressions. Incase time permits, we are planning to use the word embeddings to build a knowledge graph in order to experiment and check the benefits. All our models will be hosted using AWS/GCP.

# Proposed Plan of Research
# Stage One
# Stage Two
# Evaluation Metrics
# Preliminary Results
# References
