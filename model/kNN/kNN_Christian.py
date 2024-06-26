"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

# Modified challenge_basic to not use sklearn's kNN
# Integrated with Vishal's code

import re
import pandas as pd
import numpy as np
from collections import Counter

file_name = "clean_dataset.csv"
random_state = 42

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class kNN():

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = np.array([self.y_train[i] for i in k_indices])
        label_sum = np.sum(k_nearest_labels, axis=0)
        most_common = np.argmax(label_sum)
        return most_common

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """
    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    data = pd.read_csv(file_name)

    # Apply preprocessing to numeric fields
    data['Q7'] = data['Q7'].apply(to_numeric).fillna(0)
    data['Q8'] = data['Q8'].apply(to_numeric).fillna(0)
    data['Q9'] = data['Q9'].apply(to_numeric).fillna(0)

    # Convert Q1 to its first number
    data['Q1'] = data['Q1'].apply(get_number)
    data['Q2'] = data['Q2'].apply(get_number)
    data['Q3'] = data['Q3'].apply(get_number)
    data['Q4'] = data['Q4'].apply(get_number)

    # Process Q6 to create area rank categories
    data['Q6'] = data['Q6'].apply(get_number_list_clean)

    temp_names = []
    for i in range(1, 7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
    del data["Q6"]

    # Create category indicators and dummy variables
    new_names = []
    for col in ["Q1", "Q2", "Q3", "Q4", "Q8", "Q9"] + temp_names:
        indicators = pd.get_dummies(data[col], prefix=col)
        new_names.extend(indicators.columns)
        data = pd.concat([data, indicators], axis=1)
        del data[col]

    # Create multi-category indicators
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5_{cat}"
        new_names.append(cat_name)
        data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    del data["Q5"]


    # Preparing the features and labels
    data = data[new_names + ["Q7", "Label"]]
    data = data.sample(frac=1, random_state=42)

    x = data.drop("Label", axis=1).values
    y = pd.get_dummies(data["Label"]).values

    n_train = 1200

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # Train and evaluate classifiers
    knn = kNN(k=5)
    knn.fit(x_train, y_train)
    train_predictions = knn.predict(x_train)
    test_predictions = knn.predict(x_test)

    train_acc = np.mean(np.argmax(y_train, axis=1) == train_predictions)
    test_acc = np.mean(np.argmax(y_test, axis=1) == test_predictions)

    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")