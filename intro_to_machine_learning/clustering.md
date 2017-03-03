# Clustering

This is an unsupervised machine learning technique used to find information about
an unstructured data set.

We can use a lot of different algorithms for clustering data. One of them is
`k-means clustering`. More algos are described
[here](http://scikit-learn.org/stable/modules/clustering.html).

Even if we have uniform data clustering can tell us information about
different data points. That some are a little more alike than others.



### More Info

This is a good visualization of k-means clustering:
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

### Q&A

Q. How does k-means clustering work?

A. We deploy points called centroids in the data and then work in a 2-step iterative
process until the centroids are perfectly located in the centre of the clusters.

The two steps are 1) assign and 2) optimize, where we assign the centroids to a
specific location closest to the centers of the clusters, and then optimize where
we minimize the total quadratic error between points and centroids.

---

Q. What are challenges and limitations of k-means clustering?

A. The output of each clustering may be different based on where the centroids
are initialized and you may have counterintuitive clusters created because of the
hill climbing algo we use for k-means clustering can get stuck on local minimum.


### Example

k-means clustering

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [4, 4]])

kmeans.cluster_centers_
```
