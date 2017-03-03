# K-Nearest Neighbors (KNN)

This uses the entire training dataset to make predictions for new examples. There
is no training required. Predictions for a new instance (x) are made by searching
the entire training dataset for the K most similar instances (neighbors) and
summarizing the output variable for those K instances (for classification it might
be the most common class, for regression it might take the mean).

As we increase the `K` instances in the KNN classifier we smooth the data
and make it more resistant to outliers.

![knn_visualization](knn_visualization.png)

`good`: very simple to understand and implement. It takes no time to train.

`bad`: Pay for computational cost at test time instead of train time.

### More Info

http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/

### Q&A

### Example

```
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X,y)
pred = clf.predict([1.1])

print(pred)

OUT> [0]
```
