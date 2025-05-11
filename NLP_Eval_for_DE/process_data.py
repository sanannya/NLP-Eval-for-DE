import csv
from datasets import Dataset
#-------------------------------processing input data-------------------------------#
def get_testable_data(inputfilename):
    data = []
    with open(inputfilename, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
    categories = list(data[0].keys())
    testable_data = []
    ground_truths = []
    if (len(categories)==1):
        for dictn in data:
            for key, value in dictn.items():
                testable_data.append(value)
    else:
        for dictn in data:
            for key, value in dictn.items():
                if (key==categories[0]):
                    testable_data.append(value)
                elif (key==categories[1]):
                    ground_truths.append(value)

    testable_data = [s for s in testable_data if s]
    ground_truths = [s for s in ground_truths if s]

    return [testable_data, ground_truths]

def get_codes(inputfilename):
    data = []
    codes = []
    with open(inputfilename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data.append(row)
    for dtpt in data:
        codes.append(dtpt[0])
    codes = [s for s in codes if s]
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
