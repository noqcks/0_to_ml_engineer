# SVM (Support Vector Machines)

This is a supervised classification algorithm that helps us distinguish the decision
boundary between classes of features. This boundary can be linear or non-linear,
unlike naive bayes which can only be linear.

A hyperplane is one that splits the input vector space. In SVM, a hyperplane would
split two classes of features.

The best separating line is the one that maximizes the distance to nearest point
of either classes. This is called the margin.

There are a number of parameters we can set before fitting as SVM (kernel, gamma,
C, etc). See Q&A for more information on these.

`good`: They work well in
`bad`: They don't perform well in really large datasets, because the training time
is too large. They don't work that well with lots of overlap of features (naive bayes
is better here).

### More Info

http://machinelearningmastery.com/support-vector-machines-for-machine-learning/

### Q&A

Q. Why is it called a Support Vector Machine?

A. The two closest points of each class are called the support vectors. They support
or define the hyperplane.

---

Q. How can we do non linear classification using SVM?

A. We map the original feature space to some higher-dimensional feature space
where the training set can be separated linearly in the higher-dimension using a
kernel function.

![non_linear_svm_dimensions](assets/non_linear_svm_dimensions.png)

---

Q. What is the kernel trick? Why do we need it?

A. The kernel trick increases the size of the input space so that while the decision
boundary might be non-linear in the original input space, it now becomes linear
in the higher dimensional space.

We could try linearize our data ourselves by modifying the input variables to make
it so (i.e make all our inputs |x| to turn things linear), but this takes a lot of
human effort to look and see what kind of transformations could be done. The kernel
trick takes the effort out turning non-linear decision boundaries to linear ones.

The kernel trick takes non-linear input space and maps it to a higher dimension so
that we can "cut" through linearly.

This is a good visualization: https://www.youtube.com/watch?v=3liCbRZPrZA

---

Q. What is a "soft margin" SVM?

A. This is an SVM that allows some examples to be ignored or placed on the wrong
side of the margin, which leads to an overall better fit. The parameter C is the
SVM parameter for the soft margin cost function, which controls the influence of
each individual support vector. We trade error penalty for stability.

---

Q. What kind of parameters can we use before fitting an SVM model?

A.

- C -> which is the soft margin cost function (trading error penalty for stability)
- kernel -> the type of kernel to use to create a decision boundary.
- gamma ->

### Example

```
from sklearn.svm import SVC

X = [[0,0], [1,1]]
y = [0,1]

clf = SVC(kernel="linear")
clf.fit(X,y)
clf.predict([[2.,2.]])

OUT> [1]
```
