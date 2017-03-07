#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Cleans away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).
    """

    # predictions and actual values (net worths) have smallest errors
    # to calculate error as specified in the code comment
    errors = net_worths - predictions
    cleaned_data = zip(ages, net_worths, errors)

    # the [0] isn't necessary in this case since errors is a 1-D array
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2], reverse=True)

    limit = int(len(net_worths) * 0.1)

    # cast the iterator object as a list. I needed to do this to avoid errors
    # in the calling code.
    return list(cleaned_data[limit:])
