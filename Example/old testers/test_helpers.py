import os, sys

from NLP_Eval_for_DE import helpers, process_data, results
#accessing Main files

#testing everything w/ the 3-word pain points + 103 participant inputs & their ground truths

codes = process_data.get_codes("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\pain_point_codes_desc")
#print(codes)
#print(helpers.get_testable_data("Pain points data-ground truth.txt"))

testable_data = process_data.get_testable_data("C:\\Users\\engrla\\NLP-Eval-for-DE\\Example\\Pain points data-ground truth.txt")
# print(testable_data)
# print(len(testable_data[0]))
# print(len(testable_data[1]))
# print(helpers.make_dataset(testable_data))

# print(helpers.get_BART_scores(testable_data,codes))
# print(helpers.get_BERT_scores(testable_data,codes))
# print(helpers.get_MPNet_scores(testable_data,codes))
# print(testable_data[1])
# print(helpers.get_BERT_scores(testable_data[0],codes[0]))

# predictions = helpers.get_MPNet_scores(testable_data[0], codes[0])
# print(predictions[1])
# print(helpers.evaluate(testable_data[1], predictions[0], codes[1])[0])
# print(helpers.evaluate(testable_data[1], predictions[0], codes[1])[1])
# predictions = helpers.get_Jina_scores(testable_data[0], codes[0])
# print(predictions[1])
# print(helpers.evaluate(testable_data[1], predictions[0], codes[1])[0])
# print(helpers.evaluate(testable_data[1], predictions[0], codes[1])[1])

predictions = helpers.get_BERT_scores(testable_data[0], codes[0])
print(predictions[0])