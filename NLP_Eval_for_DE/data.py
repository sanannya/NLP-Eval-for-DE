import csv
from datasets import Dataset
import pandas as pd
#-------------------------------processing input data-------------------------------#
def get_testable_data(inputfilename):
    df = pd.read_csv(inputfilename)
    return df

def get_codes(inputfilename):
    df = pd.read_csv(inputfilename, header=None, names=["Codes"])
    return df

# def make_dataset(testable_data):
#     #process into a dataset 
#     data_numbers = []
#     for i in range(len(testable_data)):
#      data_numbers.append(i)

#     data = {
#         'TD': data_numbers,
#         'text': testable_data
#     }
#     dataset = Dataset.from_dict(data)
#     return dataset

def written_code_to_numbers(datafile, codefile):
    #if your ground truths are in written instead of numbered format, use this first to get the correct formatting
    #will assign code numbers sequentially based on the codebook file
    #will create a new input data file, but the codebook file stays the same
    data = pd.read_csv(datafile)
    codes = pd.read_csv(datafile)
    numberize = lambda phrase: codes["Codes"].index[0] if phrase==codes["Codes"] else None
    data.iloc[:,1] = data.iloc[:,1].apply(numberize)
    print (data)
    # codes = get_codes(codefile)[0]
    # new_data = []

    # with open(datafile, 'r') as infile, open("new datafile.csv", 'w', newline='') as outfile:
    #     reader = csv.DictReader(infile)
    #     new_val = "0"
    #     for row in reader:
    #         categories = list(row.keys())
    #         new_row = row
    #         #print(row)
    #         for i in range(len(codes)):
    #             if (codes[i] == row[categories[1]]):
    #                 # print("match!")
    #                 # print(row[categories[1]])
    #                 # print(codes[i])
    #                 new_val = str(i+1)
    #                 #print(new_row[categories[1]])
    #         if (row[categories[1]] not in codes):
    #             new_val = "0"
    #         new_row[categories[1]] = new_val
    #         #print(new_row)
    #         new_data.append(new_row)

    #     writer = csv.DictWriter(outfile, fieldnames=categories)
    #     writer.writeheader()
    #     writer.writerows(new_data)

