#this file is for getting the info from helper functions into txt format
import sys
import csv 
from NLP_Eval_for_DE import process_data, helpers

def get_predictions(model_name, inputs_datafile, codes_datafile, outputname):
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

    new_data = []
    for i in range(len(testable_data[0])):
        new_data.append({"Data": testable_data[0][i], "Prediction": predictions[0][i]})

    filename = outputname + '.csv'
    fieldnames = ['Data', 'Prediction']

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)

def get_scores(model_name, inputs_datafile, codes_datafile, outputname):
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

        new_data = []
        for i in range(len(testable_data[0])):
            for j in range(len(codes[0])):
                new_data.append({"Data": testable_data[0][i], "Score": str(predictions[1][i][j]), 'Code': codes[0][j]})

        filename = outputname + '.csv'
        fieldnames = ['Data', 'Score', 'Code']

        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_data)

def get_evaluation(inputs_datafile, codes_datafile,):
    #evaluation is done all together, u dont get a choice other than ur dataset & codebook
    testable_data = process_data.get_testable_data(inputs_datafile)
    codes = process_data.get_codes(codes_datafile)
    BARTpredictions = helpers.get_BART_scores(testable_data[0], codes[0])
    BERTpredictions = helpers.get_BERT_scores(testable_data[0], codes[0])
    MPNetpredictions = helpers.get_MPNet_scores(testable_data[0], codes[0])
    Jinapredictions = helpers.get_Jina_scores(testable_data[0], codes[0])

    #output the matrices
    new_data = helpers.evaluate(testable_data[1], BARTpredictions[0], codes[1])[0]
    filename = "BART_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = helpers.evaluate(testable_data[1], BERTpredictions[0], codes[1])[0]
    filename = "BERT_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = helpers.evaluate(testable_data[1], MPNetpredictions[0], codes[1])[0]
    filename = "MPNet_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = helpers.evaluate(testable_data[1], Jinapredictions[0], codes[1])[0]
    filename = "Jina_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    #output the f1 scores
    all_preds = []
    all_preds.append(BARTpredictions)
    all_preds.append(BERTpredictions)
    all_preds.append(MPNetpredictions)
    all_preds.append(Jinapredictions)
    f1s = []
    for predictions in all_preds:
        f1s.append(helpers.evaluate(testable_data[1], predictions[0], codes[1])[1])

    new_data = []
    for i in range(len(f1s)):
        new_data.append({"Code": testable_data[0][i], "BART F1": f1s[0][i], "BERT F1": f1s[1][i], "MPNet F1": f1s[2][i], "Jina F1": f1s[3][i]})

    filename = "F1_scores" + '.csv'
    fieldnames = ['Code', 'BART F1', 'BERT F1', 'MPNet F1', 'Jina F1']

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)

    #output the cohen's kappas
    models = ["BART", "BERT", "MPNet", "Jina"]
    kappas = []
    for predictions in all_preds:
        kappas.append(helpers.evaluate(testable_data[1], predictions[0], codes[1])[2])

    new_data = []
    for i in range(len(kappas)):
        new_data.append({"Model": models[i], "Cohen's Kappa": kappas[i]})

    filename = "Cohens_kappas" + '.csv'
    fieldnames = ['Model', "Cohen's Kappa"]

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)