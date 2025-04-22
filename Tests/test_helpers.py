import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("helpers.py"))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("main.py"))))

from Main import helpers, main
#accessing Main files

print(helpers.get_codes("pain_point_codes_3word.txt"))