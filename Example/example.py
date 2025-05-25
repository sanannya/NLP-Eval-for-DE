from NLP_Eval_for_DE import helpers, process_data, results

results.get_predictions("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain points full.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\description codes.csv", "predictions-jina-desc-painpoints")
# results.get_scores("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc", "score_test2")

#results.get_evaluation("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain points full.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\short titles+descriptions.csv")

#process_data.written_code_to_numbers("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\data test, single GTs only.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\hackathon open codes.csv")
#results.get_evaluation("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\new datafile.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\hackathon open codes.csv")
#results.get_predictions("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\new datafile.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\hackathon open codes.csv", "hackathon cursed run")