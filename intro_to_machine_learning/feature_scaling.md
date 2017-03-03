# Feature Scaling

Feature scaling is a way to rescale features number so that they are approximately
equivalent. This is one important part of data normalization process.

For example we might want to do computations for height and weight, but these
values are fairly different (6.1 feet vs 180 lbs), so we can rescale these to be
in between 0 and 1. They still contain the same amount of information, but they
are just represented in a different way that makes them easy to do compute tasks
on (like sum).

The formula we use for feature scaling is as follows:

![feature_scaling](feature_scaling.svg)

where x(min) is the min x value in our dataset, and x(max) is the max value in our
dataset.

### Example

[This](http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range) is the min/max scaler in sklearn.

```
from sklearn.preprocessing import MinMaxScaler
import numpy as np

weights = np.array([[115], [140], [175]])

scaler = MinMaxScaler()

rescaled_weight = scaler.fit_transform(weights)

print rescaled_weight
```
