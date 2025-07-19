import pandas as pd
import numpy as np
#-------------------------------processing input data-------------------------------#
def get_testable_data(inputfilename):
    df = pd.read_csv(inputfilename)
    return df

def get_codes(inputfilename):
    df = pd.read_csv(inputfilename, header=None, names=["Codes"])
    return df

def written_code_to_numbers(datafile, codefile):
    #if your ground truths are in written instead of numbered format, use this first to get the correct formatting
    #will assign code numbers sequentially based on the codebook file
    #will create a new input data file, but the codebook file stays the same
    data = pd.read_csv(datafile)
    codes = get_codes(codefile)
    numberize = lambda phrase: (codes["Codes"].tolist().index(phrase)+1) if phrase in codes["Codes"].values else None
    data[data.columns[1]] = data[data.columns[1]].apply(numberize)
    data = data.dropna() #just in case of user error
    data[data.columns[1]] = data[data.columns[1]].astype(int)
    data.to_csv("numberized input phrases + ground truths.csv", index=False)
    print ("Conversion complete!")
