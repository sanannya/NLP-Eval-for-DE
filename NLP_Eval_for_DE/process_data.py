from datasets import Dataset
#-------------------------------processing input data-------------------------------#
def get_testable_data(inputfilename):
    #returns a 2D array containing testable data (phrases) and their ground truths (see spreadsheet formatting example)
    raw_data = open(inputfilename, "r") 
    data = raw_data.read() 

    data_list = data.split("\n")
    for i in range(len(data_list)):
        d = data_list[i].split("\t")
        data_list[i] = d

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
    for p in range(len(data_list)):
        if (type(data_list[p]) == list):
            testable_data.append(data_list[p][0])
            ground_truths.append(data_list[p][1])
        elif (type(data_list[p]) == string):
            testable_data.append(data_list[p])

    #ground truths array is empty if there are none
    return [testable_data, ground_truths]

def get_codes(inputfilename):
    #returns an array of the codes, pulled from a tab-split text file
    raw_data = open(inputfilename, "r")
    data = raw_data.read()
    codes = data.split("\n")

    #remove extra characters
    for string in codes:
        if "\n" in string:
            new_string = string.replace("\n", "")
            #print(string)
            i = codes.index(string)
            codes[i] = new_string
    for string in codes:
        if "\t" in string:
            new_string = string.replace("\t", "")
            #print(string)
            i = codes.index(string)
            codes[i] = new_string
        if '"' in string:
            new_string = string.replace('"', "")
            i = codes.index(string)
            codes[i] = new_string

    for code in codes:
        if len(code) == 0:
            codes.remove(code)

    raw_data.close() 
    return [codes, len(codes)]

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
