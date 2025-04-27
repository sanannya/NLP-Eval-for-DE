# main.py
# Import specific modules instead of using *
from NLP_Eval_for_DE import driver, helpers

# Call the functions
driver.func1()

codes = helpers.get_codes("C:\\Users\\engrla\\NLP-Eval-for-DE\\NLP_Eval_for_DE\\Tests\\pain_points_codes_3word.txt")
testable_data = helpers.get_testable_data("C:\\Users\\engrla\\NLP-Eval-for-DE\\NLP_Eval_for_DE\\Tests\\Pain points data-ground truth.txt")
predictions = helpers.get_BART_scores(testable_data[0], codes[0])
print(predictions[1])