from NLP_Eval_for_DE import helpers, process_data, results

# results.get_predictions("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc", "pred_test3")
# results.get_scores("Jina Embeddings V2", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc", "score_test2")

results.get_evaluation("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain points full.csv", "C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\short titles.csv")
# testable_data = process_data.get_testable_data("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain points full.csv")
# codes = process_data.get_codes("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\test codes data.csv")
# predictions = helpers.get_BERT_scores(testable_data[0], codes[0])
# print(helpers.evaluate(testable_data[1], predictions[0], codes[1])[1])