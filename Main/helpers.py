#---------------------processing input data------------------------
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
    return codes



def make_dataset(inputfilename, testable_data):
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

#--------------------model runners-----------------------
from transformers import pipeline

#pipeline-compatible models only, otherwise use a different runner
def run_pipeline(task, model):
    classifier = pipeline(task, model=model)
    #file open, run through pilot data, etc.. 
    #hehe