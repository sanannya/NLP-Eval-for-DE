#-------------------------------processing input data-------------------------------#
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

def get_testable_data(inputfilename):
    #returns a 2D array containing testable data (phrases) and their ground truths (see spreadsheet formatting example)
    raw_data = open(inputfilename, "r") 
    data = raw_data.read() 
    data_list = data.split("\t") 

    #remove extra characters
    for string in data_list:
        if "\n" in string:
            new_string = string.replace("\n", "")
            #print(string)
            i = data_list.index(string)
            data_list[i] = new_string
        if '"' in string:
            new_string = string.replace('"', "")
            i = data_list.index(string)
            data_list[i] = new_string

    raw_data.close() 

    testable_data = []
    ground_truths = []
    count = 0
    for string in data_list:
        if (len(string) > 2):
            testable_data.append(string)
            count += 1
        elif (len(string) > 0):
            ground_truths.append(string)

    return [testable_data, ground_truths]

def get_codes(inputfilename):
    #returns an array of the codes, pulled from a tab-split text file
    raw_data = open(inputfilename, "r")
    data = raw_data.read()
    codes = data.split("\t")

    #remove extra characters
    for string in codes:
        if "\n" in string:
            new_string = string.replace("\n", "")
            #print(string)
            i = codes.index(string)
            codes[i] = new_string
        if '"' in string:
            new_string = string.replace('"', "")
            i = codes.index(string)
            codes[i] = new_string

    raw_data.close() 
    return [codes, len[codes]]



def make_dataset(testable_data):
    #process into a dataset 
    data_numbers = []
    for i in range(len(testable_data)):
     data_numbers.append(i)

    data = {
        'TD': data_numbers,
        'text': testable_data
    }
    dataset = Dataset.from_dict(data)
    return dataset

#-------------------------------model runners-------------------------------#
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#these functions will collect score results

def get_BART_scores(datafilename, codefilename):
    all_results = []
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    testable_data = get_testable_data(datafilename)
    codes = get_codes(codefilename)
    predictions = []
    for i in range (len(testable_data)):
        max_score = 0
        idx_of_max = 0
        sequence_to_classify = KeyDataset(make_dataset(testable_data), "text")[i]
        for j in range(len(codes)):
            candidate_labels = codes[j]
            results = classifier(sequence_to_classify, candidate_labels, multi_label=False) 
            #print(str(results["scores"][0]) + "\t" + code) 
            all_results.append(results["scores"][0])
            if (results["scores"][0] > max_score):
                idx_of_max = j+1
                max_score = results["scores"][0]
        predictions.append(idx_of_max)

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]

def get_MPNET_scores(datafilename, codefilename):
    model = SentenceTransformer("all-mpnet-base-v2")
    all_results = [] 
    predictions = []
    testable_data = get_testable_data(datafilename)
    codes = get_codes(codefilename)
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
            all_results.append(results[0][j])
            if (results[0][j] > max_score):
                idx_of_max = j+1
                max_score = results[0][j]
        predictions.append(idx_of_max)

    predictions_str = []
    for pred in predictions:
        str_pred = str(pred)
        predictions_str.append(str_pred)

    return [predictions_str, all_results]

def evaluate(ground_truths, predictions, codes_length):
    labels = []
    for i in range(codes_length):
        labels.append(str(i))
    eval = f1_score(ground_truths, predictions, average=None) 
    mtx = confusion_matrix(ground_truths, predictions, labels)
    return [eval, mtx]