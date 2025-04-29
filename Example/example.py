from NLP_Eval_for_DE import helpers, process_data, results

results.get_predictions("BART", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc")
results.get_scores("BART", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc")

#write example evals next