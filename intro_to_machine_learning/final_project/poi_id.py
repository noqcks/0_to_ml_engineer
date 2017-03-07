#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV


### classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

## ------------------------------------------------------------------- ##

def add_new_features(data_dict, features_list):
    """
    Given the data dictionary of people with features, adds some features to
    """
    for name in data_dict:

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'

    return data_dict

def remove_outliers(data_dict):
    """
    This will remove outliers that I've found in the data via scatterplot
    >>>

    features = [feature1, feature2]
    data = featureFormat(data_dict, features)

    for point in data:
      feature1_data = point[0]
      feature2_data = point[1]
      plt.scatter( feature1_data, feature2_data )

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
    """
    outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E"]

    for outlier in outliers:
        data_dict.pop(outlier, 0)

    return data_dict

"""""""""
1. Features
"""""""""
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

poi = ["poi"]

## email features that we will apply PCA
features_email = [
  # "from_messages",
  "from_poi_to_this_person",
  "from_this_person_to_poi",
  # "shared_receipt_with_poi",
  # "to_messages"
]

### Financial features might have underlying features of bribe money
features_financial = [
  "bonus",
  # "deferral_payments",
  # "deferred_income",
  "director_fees",
  "exercised_stock_options",
  # "expenses",
  # "loan_advances",
  # not a lot of data points
  # "long_term_incentive",
  # "other",
  # "restricted_stock",
  # "restricted_stock_deferred",
  "salary",
  "total_payments",
  "total_stock_value"
]

features_new = [
  "poi_ratio_messages"
]

features_list = poi + features_email + features_financial + features_new


"""""""""""""""""""""
2. Remove Outliers
"""""""""""""""""""""
### Task 2: Remove outliers

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict = remove_outliers(data_dict)

"""""""""""""""
3. New Features
"""""""""""""""
### Task 3: Create new feature(s)
data_dict = add_new_features(data_dict, features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Split into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

"""""""""""""""""""""
4/5. Train & Test Classifiers
"""""""""""""""""""""
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Manual Tests
clf = GaussianNB()
# precision: 0.32, recall: 0.38
# clf = DecisionTreeClassifier()
# precision: 0.21, recall: 0.21
# clf = RandomForestClassifier()
# precision: 0.38, recall: 0.12
# clf = AdaBoostClassifier(n_estimators=20)
# precision: 0.38, recall: 0.29
# clf = LogisticRegression(C=10, random_state=42, class_weight='balanced')
# precision: 0.28, recall: 0.52

clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
