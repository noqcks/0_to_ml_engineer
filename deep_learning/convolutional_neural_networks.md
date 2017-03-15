# Convolutional Neural Networks (Covnets)

https://www.youtube.com/watch?v=ISHGyvsT0QY

These neural networks make the assumption that our input is an image which allow
us to encode certain properties into the image, so that we can reduce the number
of parameters in the network. We do this because images are highly dimensional
and regular neural nets don't scale well for them.


### CNN Dimensions
CNNs have neurons arranged in 3 dimensions - unlike a traditional NN - into width,
height, depth.

There a two different kinds of padding, `same` and `valid` padding, and a stride
length that can affect our padding.

**stride length**: is the amount by which the filter moves over an image. Increasing
the stride length decreases the dimensions (width,height) of your image/model by
reducing the number of patches that each layer observes.

**valid padding**: There is no zero padding around the image, so if our stride is 1
then we need to remove 1 from width and height from the output width/height. So an
input width/height of 28x28 would be 26x26 with a stride of 1 (since we remove 1 from
all 4 sides).

**same padding**: the output width and height are the same as the input height and width.
So an input with (28 width, 28 height) and a stride of 1, the output will be (28,28).
We add little zeros to the input image to make the sizes match.


```
new_height = (input_height - filter_height + 2 * padding)/stride + 1
new_width = (input_width - filter_width + 2 * padding)/stride + 1
```

### Parameter Calculation

We can calculate the number of parameters a CNN uses.

For example if we have the following CNN

```
input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
Zero padding of size 1 (P)
output of shape 14x14x20 (HxWxD)
```

Then the number of parameters is
```
# bias
1
# weights
(8 * 8 * 3)
# output size
(14 * 14 * 20)

# each weight is assigned to every single part of output
paramters = ((8 * 8 * 3) + 1) * (14 * 14 * 20))
parameters = 756560
```

### Pooling

Previously, we used stride to downsample and image size. But we lose a lot of
information this way. A way we can downsample without a lot of information loss
is through pooling.

We use pooling to _reduce overfitting_ and _decrease output size_. The overfitting
is reduced because their are less parameters in future layers.

They have become less favorable though for a couple reasons:
- big data; we're more concerned about underfitting
- dropout is a better regularizer
- results in a loss of information

With _max pooling_ we look at every point on the feature map, then look at some
surrounding points and then compute the maximum of all the responses around it.

```
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

`good`: parameter free. often more accurate.
`bad`: more hyperparameters (pooling size, pooling stride). more expensive to compute.

With _average pooling_ we take the average of points nearby.

### Example

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
```

