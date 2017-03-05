# Validation

Validation is what we do when we set aside x% of our data set to use as a validation
that our training set of data is actually doing it's job correctly. The test set of data
is unfit data and completely new to the algo.

### Q&A

Q. What is cross validation?

A. We can partition our dataset set into `K` different partitions (let's call them
bins). Then we can select one of the bins to use for testing, and the rest to use
for training. We run this multiple times and then average our test results over the
`K` bins.

---

Q. What's the downside to K-fold cross validation?

A. It increases the compute time, since now you're technically working with `K` variations
of your dataset, instead of just 1 variation when you simply split up your train
and test data.

---

Q. How can we use cross validation for parameter tuning?

A. We can use GridSearchCV

```
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

# a dictionary of parameters for GridSearchCV. We have possible kernels of linear
# and rbf, and a possible C value of 1 and 10.
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

# this is the algo we want to use in GridSearchCV to find the most optimal
# combination of parameters
svr = svm.SVC()

# This will create a grid of all the possible combinations of parameters (kernel and C).
clf = GridSearchCV(svr, parameters)

# This fit function tries all the parameter combinations and returns a fitted
# classifier that's automatically tuned to optimal parameter combination.
clf.fit(iris.data, iris.target)

# this prints the best parameter combination
print clf.best_params_

OUT> {'kernel': 'linear', 'C': 1}
```

### Example

http://scikit-learn.org/stable/modules/cross_validation.html

**Basic Validation**

(This will split our data into test and train sets (40% for testing))

```
from sklearn import cross_validation

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=0)
```

**K-Fold Cross Validation**

```
import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]

kf = KFold(n_splits=2)

for train, test in kf.split(X):
    print("%s %s" % (train, test))

```

