#this file is for getting the info from helper functions into txt format
import sys
from NLP_Eval_for_DE import process_data, helpers

def get_predictions(model_name, inputs_datafile, codes_datafile):
    testable_data = process_data.get_testable_data(inputs_datafile)
    codes = process_data.get_codes(codes_datafile)
    if (model_name=="BART"):
        predictions = helpers.get_BART_scores(testable_data[0], codes[0])
    if (model_name=="BERT"):
        predictions = helpers.get_BERT_scores(testable_data[0], codes[0])
    if (model_name=="MPNet"):
        predictions = helpers.get_MPNet_scores(testable_data[0], codes[0])
    if (model_name=="Jina Embeddings V2"):
        predictions = helpers.get_Jina_scores(testable_data[0], codes[0])

    #print(predictions[1])

    with open(model_name + '_results.txt', 'w') as f:
        sys.stdout = f

        for i in range(len(testable_data[0])):
            print (testable_data[0][i] + "\t" + predictions[0][i])
            print ("\n")

def get_scores(model_name, inputs_datafile, codes_datafile):
        testable_data = process_data.get_testable_data(inputs_datafile)
        codes = process_data.get_codes(codes_datafile)
        if (model_name=="BART"):
            predictions = helpers.get_BART_scores(testable_data[0], codes[0])
        if (model_name=="BERT"):
            predictions = helpers.get_BERT_scores(testable_data[0], codes[0])
        if (model_name=="MPNet"):
            predictions = helpers.get_MPNet_scores(testable_data[0], codes[0])
        if (model_name=="Jina Embeddings V2"):
            predictions = helpers.get_Jina_scores(testable_data[0], codes[0])

        with open(model_name + '_scores.txt', 'w') as f:
            sys.stdout = f

            for i in range(len(testable_data[0])):
                print ("DATA: " + testable_data[0][i])
                print ("\n")
                for j in range(len(codes[0])):
                    print(str(predictions[1][i][j]) + "\t" + codes[0][j])
                print("\n")