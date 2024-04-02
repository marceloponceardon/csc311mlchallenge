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

def encode(file_name):
    """
    This function reads the file and encodes the data in the format Vish implemented
    """
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
    return x, y

def split_data(x, y, train=0.7, test=0.15):
    """
    Split the data into training, testing and validation sets
    The default split is 70% training, 15% testing and 15% validation
    If train + test < 1, the remaining data is used for validation
    """

    n = len(x)
    n_train = int(n * train)
    n_test = int(n * test)

    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:n_train+n_test], y[n_train:n_train+n_test]
    x_valid, y_valid = x[n_train+n_test:], y[n_train+n_test:]

    return x_train, y_train, x_test, y_test, x_valid, y_valid

def main():
    """
    The main function executes the encoding and training of the model, and prints the results
    """
    # Example usage
    x, y = encode(file_name)
    print("Data encoded")
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Split 0.6 0.2
    x_train, y_train, x_test, y_test, x_valid, y_valid = split_data(x, y, 0.6, 0.2)
    print("Data split:")
    # Print in a nice format
    print(f"Training data: {x_train.shape[0]} samples")
    print(f"Testing data: {x_test.shape[0]} samples")
    print(f"Validation data: {x_valid.shape[0]} samples")

    # Train the model
    print("Model training goes here...")
    FILE_NAME = "clean_dataset.csv"
    FILE_PATH = "" # "../../model/"

    # Encode the data
    X, t = encode(FILE_PATH + FILE_NAME)
    print("Number of data points: " + str(x.shape[0]))

    # Split the data
    print("Split: 0.6 0.2 0.2")
    X_train, t_train, X_test, t_test, X_valid, t_valid = split_data(X, t, 0.6, 0.2)

    # Print in a nice format
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    print(f"Validation data: {X_valid.shape[0]} samples")

    from sklearn.linear_model import LinearRegression as LR
    from sklearn.metrics import mean_squared_error as mse

    # Fit, train and test
    linreg = LR(fit_intercept = False, copy_X = True, n_jobs = 50)
    linreg.fit(X_train, t_train)

    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)
    y_valid_pred = linreg.predict(X_valid)

    # Calculate mse for training and testing predicions
    train_mse = mse(t_train, y_train_pred)
    test_mse = mse(t_test, y_test_pred)
    valid_mse = mse(t_valid, y_valid_pred)

    print(linreg.coef_) # weights
    print("Training MSE:", train_mse)
    print("Testing MSE:", test_mse)
    print("Validation MSE:", valid_mse)

    # Print accuracies
    print("Training accuracy:", linreg.score(X_train, t_train))
    print("Testing accuracy:", linreg.score(X_test, t_test))
    print("Validation accuracy:", linreg.score(X_valid, t_valid))

if __name__ == "__main__":
    main()
