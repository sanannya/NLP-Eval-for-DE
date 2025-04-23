import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("helpers.py"))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("main.py"))))

from Main import helpers, main
#accessing Main files

#testing everything w/ the 3-word pain points + 103 participant inputs & their ground truths

codes = helpers.get_codes("pain_point_codes_3word.txt")
#print(helpers.get_testable_data("Pain points data-ground truth.txt"))

testable_data = helpers.get_testable_data("Pain points data-ground truth.txt")
#print(testable_data)
# print(helpers.make_dataset(testable_data))

# print(helpers.get_BART_scores(testable_data,codes))
# print(helpers.get_BERT_scores(testable_data,codes))
# print(helpers.get_MPNet_scores(testable_data,codes))
# print(helpers.get_Jina_scores(testable_data,codes))

predictions = helpers.get_Jina_scores(testable_data[0], codes[0])
print(helpers.evaluate(testable_data[1], predictions[0], codes[1]))