"""
Welcome! Make a copy of this file to run on your own data & codebook.

Uncomment the line you wish to run and change the parameters to use your input files.

3 options:
    - Code your data using an NLP of your choice.
    - Get similarity scores using an NLP of your choice.
    - Evaluate NLP performance on your data by comparing them against human-done coding.

Also, each function is better explained in the README.
"""
from NLP_Eval_for_DE import helpers, process_data, results

#-------------Option 1: get an NLP to code your data-------------------------------------------------------------------

#uses the Jina Embeddings V2 model to code survey data 
#results.get_predictions("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\description codes.csv", "predictions-jina-desc-painpoints")

#----------------------------------------------------------------------------------------------------------------------





#-------------Option 2: get similarity scores--------------------------------------------------------------------------

#uses the Jina Embeddings V2 model to get similarity scores between each data point compared to each code 
#results.get_scores("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\inputs\\case study 1 input\\descriptions codes.csv", "scores-jina-desc-painpoints")

#----------------------------------------------------------------------------------------------------------------------






#-------------Option 3: evaluate---------------------------------------------------------------------------------------

#gets evaluation data for survey data, comparing it to human-done codes
#results.get_evaluation("Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\inputs\\case study 1 input\\description codes.csv")

#If the human-done codes are in written form, run this first to convert them to numerical codes (required formatting)
#process_data.written_code_to_numbers("Example\\inputs\\case study 2 input-axial codes\\axial data.csv", "Example\\inputs\\case study 2 input-axial codes\\axial codes.csv")

#----------------------------------------------------------------------------------------------------------------------