# NLP-Eval-for-DE

This repository contains code to perform natural language processing AI-driven qualitative coding for design engineering. This has been employed in this paper in case studies 1 and 2.
>*******************PAPER HEREEEEEEEEEEEEEEE*******************
>Anannya Sathaye, Elisa Koolman, Anastasia M. Schauer
>***********WHERE PUBLISHED HERE*******************

Bitex citation:
```bibtex
*****PUT CITATION HERE*****
```

## Description

NLP Eval for DE is intended to perform qualitative coding of design engineering research data with a codebook, utilizing your choice of natural language processing AI model. There are two primary uses of this module:
- Qualitative Coding: you can upload input data and a codebook, then run one of the provided NLP models to get an assigned code for each data point. This is intended to complete the task so that design researchers won't have to. Our paper goes in depth on the relative accuracy of these models, as well as what parameters of the dataset & codebook may affect that accuracy.
- Evaluation: you can also test all supported NLP models on your data/codebook. This will generate confusion matrices, cohen's kappas, and F1 scores.

This README will contain code and formatting instructions. 

In the "Example" folder, you can also view the example python script and input data file to write your code and format your input files.

## Getting Started

### Dependencies

- ****LIBRARY/PACKAGE LISTED HERE, MAKE A REQURIEMENTS.TXT****

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
    - codebooks (this case study compared 3 codebooks): "description codes.csv", "short titles.csv", "short titles+descriptions.csv"
    - results: ******LIST THEM HEREEE******
- Case study 2: refer to these for formatting
    - codebook: "hackathon data + GTs.csv".
        - For running evaluations, ground truths should be in numerical format. If they're in written format (like this file), the function written_code_to_numbers will convert them to numbers, as seen in "hackathon numerical GTs.csv".
    - results: ******LIST THEM HEREEE******


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
results.get_predictions("Jina Embeddings V2", "Example\\pain points full.csv", "Example\\description codes.csv", "predictions-jina-desc-painpoints")
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
results.get_scores("Jina Embeddings V2", "Example\\pain points full.csv", "Example\\description codes.csv", "scores-jina-desc-painpoints")
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
results.get_evaluation("Example\\pain points full.csv", "Example\\short titles+descriptions.csv")
```

*******FINISH THIS W/ WRITTEN TO NUM CONVERSION, &(INCLUDE HELPER FUNC(?))************

### Formatting input files

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

- 0.1
    - Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)