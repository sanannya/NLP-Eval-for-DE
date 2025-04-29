#-------------------------------model runners-------------------------------#
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from transformers import AutoModel
from numpy.linalg import norm

from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from NLP_Eval_for_DE.process_data import make_dataset
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

#these functions will run the models, return [model predictions, similarity scores in order of the codes]

def get_BART_scores(testable_data, codes):
    all_results = []
    results = []
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    predictions = []
    record = []
    for i in range (len(testable_data)):
        max_score = 0
        idx_of_max = 0
        sequence_to_classify = KeyDataset(make_dataset(testable_data), "text")[i]
        for j in range(len(codes)):
            candidate_labels = codes[j]
            results = classifier(sequence_to_classify, candidate_labels, multi_label=False) 
            #print(str(results["scores"][0]) + "\t" + code) 
            record.append(results["scores"][0])
            if (results["scores"][0] > max_score):
                idx_of_max = j+1
                max_score = results["scores"][0]
        predictions.append(idx_of_max)
        all_results.append(record)
        record = []

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]

def get_BERT_scores(testable_data, codes):
    #this is a sentence transformers fine tuned version of BERT trained on various databases
    #https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1
    model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1")
    predictions = []
    all_results = []
    record = []
    for i in range(len(testable_data)): 
        max_score = 0
        idx_of_max = 0
        #user_embeddings = model.encode(testable_data[i])
        user_embeddings = model.encode(KeyDataset(make_dataset(testable_data), "text")[i])
        code_embeddings = model.encode(codes)
        similarities = model.similarity(user_embeddings, code_embeddings)
        results = similarities.tolist()
        for j in range(len(codes)): 
            #print(results)
            record.append(results[0][j])
            if (results[0][j] > max_score):
                idx_of_max = j+1
                max_score = results[0][j]
        predictions.append(idx_of_max)
        all_results.append(record)
        record = []

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]
    
def get_MPNet_scores(testable_data, codes):
    model = SentenceTransformer("all-mpnet-base-v2")
    all_results = [] 
    predictions = []
    record = []
    for i in range(len(testable_data)): 
        max_score = 0
        idx_of_max = 0
        #user_embeddings = model.encode(testable_data[i])
        #^^old version, replaced w/ dataset access to make eval fair 
        user_embeddings = model.encode(KeyDataset(make_dataset(testable_data), "text")[i])
        code_embeddings = model.encode(codes)
        similarities = model.similarity(user_embeddings, code_embeddings)
        results = similarities.tolist()
        for j in range(len(codes)): 
            record.append(results[0][j])
            if (results[0][j] > max_score):
                idx_of_max = j+1
                max_score = results[0][j]
        predictions.append(idx_of_max)
        all_results.append(record)
        record = []

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]

def get_Jina_scores(testable_data, codes):
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
    code_embeddings = model.encode(codes)
    all_results = [] 
    predictions = []
    results = []
    for i in range(len(testable_data)): 
        max_score = 0
        idx_of_max = 0
        user_embeddings = model.encode(KeyDataset(make_dataset(testable_data), "text")[i])
        for k in range(len(codes)):
            results.append(cos_sim(user_embeddings, code_embeddings[k]))
        for j in range(len(codes)): 
            #print(results)
            #all_results.append(results[j])
            if (results[j] > max_score):
                idx_of_max = j+1
                max_score = results[j]
        predictions.append(idx_of_max)
        all_results.append(results)
        results = []

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]

def evaluate(ground_truths, predictions, codes_length):
    labels_nums = []
    for i in range(codes_length):
        labels_nums.append(str(i))
    labels_nums.append(str(codes_length))
    eval = f1_score(ground_truths, predictions, average=None) 
    mtx = confusion_matrix(ground_truths, predictions, labels=labels_nums)
    #return correct/incorrect per code
    return [eval, mtx]