"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd

file_name = "clean_dataset.csv"
random_state = 42

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

def create_dummies_with_all_categories(series, prefix, all_categories):
    dummies = pd.get_dummies(series, prefix=prefix)
    expected_columns = [f"{prefix}_{cat}" for cat in all_categories]
    dummies = dummies.reindex(columns=expected_columns, fill_value=0)
    return dummies

def normalize_column(data, column_name):
    """Normalizes the specified column in the DataFrame.
    """
    data[column_name] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())

def get_data():

    data = pd.read_csv(file_name)

    all_possible_q1_q4_categories = [-1, 1, 2, 3, 4, 5]
    all_possible_q6_ranks = [-1, 1, 2, 3, 4, 5, 6]

    # Apply preprocessing to numeric fields
    data['Q7'] = data['Q7'].apply(to_numeric).fillna(0)
    data['Q8'] = data['Q8'].apply(to_numeric).fillna(0)
    data['Q9'] = data['Q9'].apply(to_numeric).fillna(0)

    # normalize_column(data, 'Q7')
    # normalize_column(data, 'Q8')
    # normalize_column(data, 'Q9')

    # Convert Q1 to its first number
    data['Q1'] = data['Q1'].apply(get_number)
    data['Q2'] = data['Q2'].apply(get_number)
    data['Q3'] = data['Q3'].apply(get_number)
    data['Q4'] = data['Q4'].apply(get_number)

    # Process Q6 to create area rank categories
    data['Q6'] = data['Q6'].apply(get_number_list_clean)

    for i in range(1, 7):
        col_name = f"rank_{i}"
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
        dummies = create_dummies_with_all_categories(data[col_name], col_name, all_possible_q6_ranks)
        del data[col_name]
        data = pd.concat([data, dummies], axis=1)
    del data["Q6"]


    for col in ["Q1", "Q2", "Q3", "Q4"]:
        dummies = create_dummies_with_all_categories(data[col], col, all_possible_q1_q4_categories)
        data = pd.concat([data, dummies], axis=1)
        del data[col]

    new_names = []
    # Create multi-category indicators
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5_{cat}"
        new_names.append(cat_name)
        data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    del data["Q5"]


    # Preparing the features and labels
    data = data[new_names + [col for col in data.columns if col.startswith(('Q1_', 'Q2_', 'Q3_', 'Q4_', 'Q7', 'Q8', 'Q9', 'rank_', 'Label'))]]
    data = data.sample(frac=1, random_state=42)

    #print(list(data.columns))
    x = data.drop("Label", axis=1).values
    y = pd.get_dummies(data["Label"]).values

    n_train = 1200

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    return x_train, y_train, x_test, y_test

def get_file_data(f_name):

    data = pd.read_csv(f_name)

    all_possible_q1_q4_categories = [-1, 1, 2, 3, 4, 5]
    all_possible_q6_ranks = [-1, 1, 2, 3, 4, 5, 6]
    # Apply preprocessing to numeric fields
    data['Q7'] = data['Q7'].apply(to_numeric).fillna(0)
    data['Q8'] = data['Q8'].apply(to_numeric).fillna(0)
    data['Q9'] = data['Q9'].apply(to_numeric).fillna(0)

    # normalize_column(data, 'Q7')
    # normalize_column(data, 'Q8')
    # normalize_column(data, 'Q9')

    # Convert Q1 to its first number
    data['Q1'] = data['Q1'].apply(get_number)
    data['Q2'] = data['Q2'].apply(get_number)
    data['Q3'] = data['Q3'].apply(get_number)
    data['Q4'] = data['Q4'].apply(get_number)

    # Process Q6 to create area rank categories
    data['Q6'] = data['Q6'].apply(get_number_list_clean)


    for i in range(1, 7):
        col_name = f"rank_{i}"
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
        dummies = create_dummies_with_all_categories(data[col_name], col_name, all_possible_q6_ranks)
        del data[col_name]
        data = pd.concat([data, dummies], axis=1)
    del data["Q6"]

    for col in ["Q1", "Q2", "Q3", "Q4"]:
        dummies = create_dummies_with_all_categories(data[col], col, all_possible_q1_q4_categories)
        data = pd.concat([data, dummies], axis=1)
        del data[col]

    new_names = []
    # Create multi-category indicators
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5_{cat}"
        new_names.append(cat_name)
        data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    del data["Q5"]


    # Preparing the features and labels
    data = data[new_names + [col for col in data.columns if col.startswith(('Q1_', 'Q2_', 'Q3_', 'Q4_', 'Q7', 'Q8', 'Q9', 'rank_'))]]
    data = data.sample(frac=1, random_state=42)
    
    column_ordering = None


    x = data.values
   #print((f"Len: {len(x[0])} and features: {list(data.columns)}"))
    # if len(x[0] != 73):
    #     raise Exception(f"Len: {len(x[0])} and features: {list(data.columns)}")

    return x 

#get_file_data("./clean_dataset.csv")