#!/usr/local/bin/python2

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

