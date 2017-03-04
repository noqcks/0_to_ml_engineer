# Feature Selection

The new feature selection follows a process

1. use your human intuition
2. code up a new feature
3. visualize
4. repeat

### Q&A

Q. How does bias-variance dilemma relate to feature selection?

A. As we increase the size of features set we're using, we can sometimes increase
the variance of our model, and as we decrease our feature set we increase the bias
of our model.

---

Q. What is regularization?

A. It is an automatic form of feature selection that helps us mathematically find
the best bias-variance tradeoff in our models. It automatically penalizes the extra
features that you use in your model.

---

Q. What is a lasso regression?

A. Is it an algo that we use to find optimal features. It tries to optimize the
coefficients for our linear models so that we produce the best fit for our data.
We find the the best possible coefficients through coordinate descent.

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html


### Example

lasso regression

```
from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])

print(clf.coef_)

print(clf.intercept_)
```
