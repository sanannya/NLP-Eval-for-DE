# NLP-Eval-for-DE

This repository contains code to perform natural language processing AI-driven qualitative coding for design engineering. This has been employed in this paper in case studies 1 and 2.
>*******************PAPER TITLE IS A WIP*******************
>Anannya Sathaye, Elisa Koolman, Anastasia M. Schauer
>***********WHERE PUBLISHED TBD*******************

Bitex citation:
```bibtex
*****CITATION TO BE ADDED LATER*****
```

## Description

NLP Eval for DE is intended to perform qualitative coding of design engineering research data with a codebook, utilizing your choice of natural language processing AI model. There are two primary uses of this module:
- Qualitative Coding: you can upload input data and a codebook, then run one of the provided NLP models to get an assigned code for each data point. This is intended to complete the task so that design researchers won't have to. Our paper goes in depth on the relative accuracy of these models, as well as what parameters of the dataset & codebook may affect that accuracy.
- Evaluation: you can also test all supported NLP models on your data/codebook. This will generate confusion matrices, cohen's kappas, and F1 scores.

This README will contain code and formatting instructions. 

In the "Example" folder, you can also view the example python script and input data file to write your code and format your input files.

## Getting Started

### Dependencies

Install packages using requirements.txt:
- In the command line, cd into this repo (NLP-Eval-for-DE) and run this command.
```
pip install -r requirements.txt
```

### Installing

Install from source: 
```bash
git clone https://github.com/sanannya/NLP-Eval-for-DE.git
cd NLP-Eval-for-DE
pip install -e .
```
Or clone to your local machine however you prefer. 

## Using the module

### Example files

The repository's "Example" folder contains the runnable script, input files, and output files for both case studies documented in the paper.
- example.py: you can copy and edit this file for your own use.
- Case study 1: refer to these for formatting
    - input data: pain points full.csv
    - codebooks (this case study compared 3 codebooks): "description codes.csv", "short titles.csv", "short titles+descriptions.csv"
    - results: code similarities for all 3 codebooks, evaluation results for all 3 codebooks
- Case study 2: refer to these for formatting
    - input data: "hackathon data + GTs.csv" or the input data files for axial code groupings
        - For running evaluations, ground truths should be in numerical format. If they're in written format (like this file), the function written_code_to_numbers will convert them to numbers, as seen in "hackathon numerical GTs.csv".
    - Codebooks: open codes, axial codes, and open codes grouped by axial codes
    - results: code similarities for all codebooks, evaluation for all codebooks
- extra: "predictions-jina-dec-painpoints.csv", example using this module for coding instead of evaluation.

The functions are explained in further detail in the section below.

### Executing program

- To access the functions, create a python file. Run this code as an imported module:
```
from NLP_Eval_for_DE import results
```
- Get qualitative coding results for a data set.
    - Inputs
        - Choose an NLP model to use: "BART", "BERT", "MPNet", "Jina Embeddings V2"
        - Input data file: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
        - Codebook: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
        - Choose what to name the output CSV file.
    - Outputs
        - A CSV containing the input data points and their NLP-assigned codes according to your inputted codebook
```
#example
results.get_predictions("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\description codes.csv", "predictions-jina-desc-painpoints")
```

- Get similarity scores between each data point and code.
    - Inputs
        - Choose an NLP model to use: "BART", "BERT", "MPNet", "Jina Embeddings V2"
        - Input data file: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
        - Codebook: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
        - Choose what to name the output CSV file.
    - Outputs
        - A CSV containing the input data points and their NLP-assigned similarity scores to each code. Ordered by data point, and scores displayed in the order of the codebook.
```
#example
results.get_scores("Jina Embeddings V2", "Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\inputs\\case study 1 input\\descriptions codes.csv", "scores-jina-desc-painpoints")
```

- Get evaluation results to see how all the models perform on your dataset & codebook. This will only work if your dataset has assigned ground truths to compare against the models.
    - Inputs
        - Input data file: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
        - Codebook: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
    - Outputs
        - 4 matrices: CSV files (BART_matrix, BERT_matrix, MPNet_matrix, & Jina_matrix). These are the confusion matrices comparing the ground truth codes (rows) vs model-predicted codes (columns)
        - F1_scores.csv: For each code, shows each model's F1 score (an accuracy metric). 
        - Cohens_kappas.csv: Shows each model's cohen's kappa score.
```
#example
results.get_evaluation("Example\\inputs\\case study 1 input\\pain points full.csv", "Example\\inputs\\case study 1 input\\description codes.csv")
```

- If you want to run an evaluation on your dataset, but your human-done codes are in written form (instead of the desired number form), use this function to get a compatible file.
    - Inputs:
        - Input data file: a CSV file. The written human-done codes must match the codebook exactly in spelling, upper/lowercase, and punctuation.
        - Codebook: a CSV file, must be properly formatted; see CSV formatting instructions in the next section
    - Outputs:
        - "new datafile.csv": a new input data file exactly the same as your first input parameter csv file, but the written ground truths have been changed to numbers corresponding to their location in the codebook.
```
#example
process_data.written_code_to_numbers("Example\\inputs\\case study 2 input-axial codes\\axial data.csv", "Example\\inputs\\case study 2 input-axial codes\\axial codes.csv")
```

- To help you with codebook writing or with analyzing it, run this function to get the average similarity of each code to every other code in the codebook. (This is useful for determining if a code is unique/particular enough to be a useful categorization for qualitative analysis.)
    - Input: A codebook (a CSV file, must be properly formatted; see CSV formatting instructions in the next section)
    - Output: CSV file "code similarity scores.csv" containing each code & its average similarity score between itself and all the other codes.
```
#example
results.get_similarity_scores("Example\\inputs\\case study 1 input\\short titles+descriptions.csv")
```

### Formatting input files
Instructions for formatting input files are detailed here. You can also see the example input files in this repo's Example folder (Example->inputs). They are organized by case study, with each sub-fodler containing codebook(s) and participant input data files.

- Input data: this is the CSV containing the data you want to code. 
    - Example: Example->inputs->case study 1 input->"pain points full.csv"
    - Column 1 (heading can be whatever you want) is the input data
    - Column 2 (optional, heading can be whatever you want) is the human-done codes
        - only needed if you want to run evaluation
        - IMPORTANT: this column's values MUST be numerical. If they are in written form, use the written_code_to_numbers function to convert them to numbers
        - if you just want to code the data, you can omit having a second column, get_predictions will still work.

- Codebook: this CSV contains the codes in the codebook
    - Example: Example->inputs->case study 1 input->"description codes.csv".
    - Column 1: no heading. List the codes in this column.
    - Note: if running written_code_to_numbers with this file, spelling/case must match that of the human done written codes.

## Authors

- Code by Anannya Sathaye. 
    - Contact: anannya.sathaye@utexas.edu to let me know about issues or for questions.
- NLP testing methodology developed by Anannya Sathaye, Elisa Koolman, and Anastasia Schauer

## Version History

- 0.1
    - Initial Release
