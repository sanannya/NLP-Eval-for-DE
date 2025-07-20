#-------------------------------model runners-------------------------------#
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from transformers import AutoModel
from numpy.linalg import norm

import pandas as pd
import numpy as np

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

#these functions will run the models

def get_BART_scores(testable_data_df, codebook_df):
    results = []
    classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
    predictions = []
    record = []

    all_results = pd.DataFrame(columns=["Input phrase", "Similarity scores"])
    predictions = pd.DataFrame(columns=["Input phrase", "Assigned code"])

    for i, row1 in testable_data_df.iterrows():
        max_score = 0
        idx_of_max = 0
        sequence_to_classify = row1.iloc[0]
        for j, row2 in codebook_df.iterrows():
            candidate_labels = row2.iloc[0]
            results = classifier(sequence_to_classify, candidate_labels, multi_label=False) 
            record.append(results["scores"][0])
            if (results["scores"][0] > max_score):
                idx_of_max = j+1
                max_score = results["scores"][0]
        predictions.loc[len(predictions)] = [sequence_to_classify, idx_of_max]
        all_results.loc[len(all_results)] = [sequence_to_classify, record]
        record = []

    return [predictions, all_results]

def get_BERT_scores(testable_data_df, codebook_df):
    #this is a sentence transformers fine tuned version of BERT trained on various databases
    #https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1
    model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1")
    all_results = pd.DataFrame(columns=["Input phrase", "Similarity scores"])
    predictions = pd.DataFrame(columns=["Input phrase", "Assigned code"])
    code_embeddings = model.encode(codebook_df["Codes"].tolist())
    record = []
    for i, row in testable_data_df.iterrows(): 
        max_score = 0
        idx_of_max = 0
        user_embeddings = model.encode(row.iloc[0])
        similarities = model.similarity(user_embeddings, code_embeddings)
        results = similarities.tolist()
        for j in range(len(codebook_df["Codes"].tolist())):
            record.append(results[0][j])
            if (results[0][j] > max_score):
                idx_of_max = j+1
                max_score = results[0][j]
        predictions.loc[len(predictions)] = [row.iloc[0], idx_of_max]
        all_results.loc[len(all_results)] = [row.iloc[0], record]
        record = []

    return [predictions, all_results]
    
def get_MPNet_scores(testable_data_df, codebook_df):
    model = SentenceTransformer("all-mpnet-base-v2")
    all_results = pd.DataFrame(columns=["Input phrase", "Similarity scores"])
    predictions = pd.DataFrame(columns=["Input phrase", "Assigned code"])
    code_embeddings = model.encode(codebook_df["Codes"].tolist())
    record = []
    for i, row in testable_data_df.iterrows(): 
        max_score = 0
        idx_of_max = 0
        user_embeddings = model.encode(row.iloc[0])
        similarities = model.similarity(user_embeddings, code_embeddings)
        results = similarities.tolist()
        for j in range(len(codebook_df["Codes"].tolist())): 
            record.append(results[0][j])
            if (results[0][j] > max_score):
                idx_of_max = j+1
                max_score = results[0][j]
        predictions.loc[len(predictions)] = [row.iloc[0], idx_of_max]
        all_results.loc[len(all_results)] = [row.iloc[0], record]
        record = []

    return [predictions, all_results]

def get_Jina_scores(testable_data_df, codebook_df):
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
    code_embeddings = model.encode(codebook_df["Codes"].tolist())
    all_results = pd.DataFrame(columns=["Input phrase", "Similarity scores"])
    predictions = pd.DataFrame(columns=["Input phrase", "Assigned code"])
    record = []
    for i, row in testable_data_df.iterrows(): 
        max_score = 0
        idx_of_max = 0
        user_embeddings = model.encode(row.iloc[0])
        for k in range(len(codebook_df["Codes"].tolist())):
            record.append(cos_sim(user_embeddings, code_embeddings[k]))
        for j in range(len(codebook_df["Codes"].tolist())): 
            if (record[j] > max_score):
                idx_of_max = j+1
                max_score = record[j]
        predictions.loc[len(predictions)] = [row.iloc[0], idx_of_max]
        all_results.loc[len(all_results)] = [row.iloc[0], record]
        record = []

    return [predictions, all_results]

def evaluate(ground_truths, predictions, codes_length):
    labels_nums = []
    for i in range(codes_length):
        labels_nums.append(str(i))
    labels_nums.append(str(codes_length))
    f1s = f1_score(ground_truths, predictions, labels=labels_nums, average=None, zero_division=0.0) 
    mtx = confusion_matrix(ground_truths, predictions, labels=labels_nums)
    kappa = cohen_kappa_score(ground_truths, predictions, labels=labels_nums, weights=None, sample_weight=None)
    #return correct/incorrect per code
    return [mtx, f1s, kappa]