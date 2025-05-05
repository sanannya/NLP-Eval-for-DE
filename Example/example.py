from NLP_Eval_for_DE import helpers, process_data, results

results.get_predictions("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc", "test1")
results.get_scores("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc", "test2")

#write example evals next