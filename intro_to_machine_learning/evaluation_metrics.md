# Evaluation Metrics

accuracy = (num items in a class labeled correctly) / (all items in that class)
recall = (true positives) / (true positives + false negatives)
precision = (true positives) / (true positives + false positives)


### More Info

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

F1 Score
https://en.wikipedia.org/wiki/F1_score

### Q&A

Q. What is a confusion matrix?

A. A confusion matrix is a tally of all the points that were correctly or incorrectly
classified by a supervised ml algo.

---

Q. What is true positive, true negatives, false negatives and false positives?

A.

This is a comparison of actual vs predicted.

```
true positive = predicted as true; actually true.
false positive = predicted as true; actually false.

true negative = predicted as false; actually false.
false negative = predicted as false; actually true.
```

---

Q. What is recall and precision for a confusion matrix?

A. They are both based on an understanding of relevance.

high precision = more relevant results than irrelevant ones
high recall = an algo returned most of the relevant results

```
# fraction of retrieved instances that are relevant
precision = (true positives) / (true positives + false positives)
# fraction of relevant instances that are retrieved
recall = (true positives) / (true positives + false negatives)
```

![recall and precision](assets/recall_and_precision.png)

### Example

```
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print recall_score(true_labels, predictions)
print precision_score(true_labels, predictions)
```
