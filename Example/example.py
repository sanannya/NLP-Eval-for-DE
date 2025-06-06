"""
Welcome! Make a copy of this file to run on your own data & codebook.

Uncomment the line you wish to run and change the parameters to use your input files.

4 options:
    - Code your data using an NLP of your choice.
    - Get similarity scores using an NLP of your choice.
    - Evaluate NLP performance on your data by comparing them against human-done coding.
    - Evaluate your codebook by getting the average similarity of each code to every other code.

Also, each function is better explained in the README.
"""
from NLP_Eval_for_DE import helpers, process_data, results
import spacy

'''-------------Option 1: get an NLP to code your data-------------------------------------------------------------------'''

'''uses the Jina Embeddings V2 model to code survey data'''
#results.get_predictions("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\description codes.csv", "predictions-jina-desc-painpoints")

'''----------------------------------------------------------------------------------------------------------------------'''





'''-------------Option 2: get similarity scores--------------------------------------------------------------------------'''

'''uses the Jina Embeddings V2 model to get similarity scores between each data point compared to each code '''
#results.get_scores("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\inputs\\case study 1 input\\descriptions codes.csv", "scores-jina-desc-painpoints")

'''----------------------------------------------------------------------------------------------------------------------'''






'''-------------Option 3: evaluate---------------------------------------------------------------------------------------'''

'''gets evaluation data for survey data, comparing it to human-done codes'''
#results.get_evaluation("Example\\inputs\\case study 2 input-grouped by axials\\Individual goal numerical.csv", "Example\\inputs\\case study 2 input-grouped by axials\\Individual goal codes.csv")

'''If the human-done codes are in written form, run this first to convert them to numerical codes (required formatting)'''
#process_data.written_code_to_numbers("Example\\inputs\\case study 2 input-axial codes\\axial data.csv", "Example\\inputs\\case study 2 input-axial codes\\axial codes.csv")

'''----------------------------------------------------------------------------------------------------------------------'''






'''-------------Option 4: codebook similarity---------------------------------------------------------------------------------------'''

'''run this once to download the spaCy medium english language model 
(you may also download the large model "en_core_web_lg", but the medium & large models have similar accuracy)'''
#spacy.cli.download("en_core_web_md")

'''generate a csv file containing each code & its average similarity score'''
results.get_similarity_scores("Example\\inputs\\case study 1 input\\short titles+descriptions.csv")