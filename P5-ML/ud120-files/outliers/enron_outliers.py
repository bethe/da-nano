# coding: utf-8
# %load "../ud120-files/outliers/enron_outliers.py"
#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../ud120-files/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../ud120-files/final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

### Remove 'Total' line
data_dict.pop('TOTAL')

data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print data.max()
# %save "../ud120-files/outliers/enron_outliers.py"
