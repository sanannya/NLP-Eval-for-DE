#this file is for getting the info from helper functions into txt format
import csv
import pandas as pd
import spacy
from NLP_Eval_for_DE import data, scores

def get_predictions(model_name, inputs_datafile, codes_datafile, outputname):
    testable_data = data.get_testable_data(inputs_datafile)
    codes = data.get_codes(codes_datafile)
    if (model_name=="BART"):
        predictions = scores.get_BART_scores(testable_data, codes)
    if (model_name=="BERT"):
        predictions = scores.get_BERT_scores(testable_data, codes)
    if (model_name=="MPNet"):
        predictions = scores.get_MPNet_scores(testable_data, codes)
    if (model_name=="Jina Embeddings V2"):
        predictions = scores.get_Jina_scores(testable_data, codes)

    predictions[0].to_csv(outputname+".csv", index=False)

def get_scores(model_name, inputs_datafile, codes_datafile, outputname):
        testable_data = data.get_testable_data(inputs_datafile)
        codes = data.get_codes(codes_datafile)
        if (model_name=="BART"):
            all_scores = scores.get_BART_scores(testable_data, codes)[1]
        if (model_name=="BERT"):
            all_scores = scores.get_BERT_scores(testable_data, codes)[1]
        if (model_name=="MPNet"):
            all_scores = scores.get_MPNet_scores(testable_data, codes)[1]
        if (model_name=="Jina Embeddings V2"):
            all_scores = scores.get_Jina_scores(testable_data, codes)[1]

        all_scores.iloc[:, 1] = all_scores.iloc[:, 1].apply(lambda x: list(map(float, x)))
        all_scores.to_csv(outputname+".csv", index=False)

def get_evaluation(inputs_datafile, codes_datafile):
    testable_data = data.get_testable_data(inputs_datafile)
    codes = data.get_codes(codes_datafile)
    BARTpredictions = scores.get_BART_scores(testable_data, codes)
    BERTpredictions = scores.get_BERT_scores(testable_data, codes)
    MPNetpredictions = scores.get_MPNet_scores(testable_data, codes)
    Jinapredictions = scores.get_Jina_scores(testable_data, codes)

    #output the matrices
    new_data = scores.evaluate(testable_data, BARTpredictions[0], codes)[0]
    filename = "BART_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = scores.evaluate(testable_data, BERTpredictions[0], codes)[0]
    filename = "BERT_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = scores.evaluate(testable_data, MPNetpredictions[0], codes)[0]
    filename = "MPNet_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    new_data = scores.evaluate(testable_data, Jinapredictions[0], codes)[0]
    filename = "Jina_matrix" + '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_data)

    #output the f1 scores
    
    codes = codes["Codes"].tolist()
    all_preds = []
    all_preds.append(BARTpredictions)
    all_preds.append(BERTpredictions)
    all_preds.append(MPNetpredictions)
    all_preds.append(Jinapredictions)
    f1s = []
    for predictions in all_preds:
        f1s.append(scores.evaluate(testable_data, predictions[0], codes)[1])

    new_data = []
    codes.insert(0, "No code")
    for i in range(len(codes)):
        new_data.append({"Code": codes[i], "BART F1": f1s[0][i], "BERT F1": f1s[1][i], "MPNet F1": f1s[2][i], "Jina F1": f1s[3][i]})

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
        kappas.append(scores.evaluate(testable_data, predictions[0], codes)[2])

    new_data = []
    for i in range(len(kappas)):
        new_data.append({"Model": models[i], "Cohen's Kappa": kappas[i]})

    filename = "Cohens_kappas" + '.csv'
    fieldnames = ['Model', "Cohen's Kappa"]

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)

def get_similarity_scores(codes_datafile):
    nlp = spacy.load("en_core_web_md")

    new_data = []
    codes = data.get_codes(codes_datafile)
    codes = codes["Codes"].to_list()
    for i in range(len(codes)):
        sum = 0
        for j in range(len(codes)):
            if (i!=j):
                doc1 = nlp(codes[i])
                doc2 = nlp(codes[j])
                sum += doc1.similarity(doc2)
        sum = sum/(len(codes)-1)
        new_data.append({"Code": codes[i], "Average similarity to other codes": sum})
        
    filename = "code similarity scores" + '.csv'
    fieldnames = ['Code', 'Average similarity to other codes']
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_data)