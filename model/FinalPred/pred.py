import os
import numpy as np
from dataParsing import get_data
import csv


class MLPModel(object):
    def __init__(self, num_features=138, num_hidden=(300, 300, 300, 300), num_classes=4, activation="logistic"):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = num_features
        self.layer_format = num_hidden
        self.num_classes = num_classes
        self.num_layers = len(num_hidden) + 1
        self.activation = activation

        # Array of all the weight matrices for each layer
        self.layer_matrices = []

        # Add First Layer
        self.layer_matrices.append(np.zeros((self.layer_format[0], self.num_features)))

        # Add deep layers
        for i in range(self.num_layers - 2):
            self.layer_matrices.append(np.zeros((self.layer_format[i+1], self.layer_format[i])))

        # Last hidden layer to classes layer
        self.layer_matrices.append(np.zeros((self.num_classes, self.layer_format[-1])))



        # Read weights from files and set it into matrices
        # Dont even try to dechiper what I wrote here
        for i in range(self.num_layers):
            weight_matrix = self.layer_matrices[i]
            with open(f"./Weights/Layer{i}weights.txt", "r") as f:
                weight_col_num = 0
                col_weights = []
                for line in f.readlines():
                    if "Node " in line:
                        if col_weights:
                            assert len(col_weights) == weight_matrix.shape[0]
                            weight_matrix[:,weight_col_num] = col_weights
                            col_weights = []
                        node_num = line[len("Node "):-3]
                        weight_col_num = int(node_num) 
                    else:
                        line = line.strip()
                        if line.find("[") != -1:
                            line = line[1:].strip()
                        if line.find("]") != -1:
                            line = line[:-1].strip()
                        nums = [float(n.strip()) for n in line.split(" ") if n]
                        col_weights.extend(nums)
                


    def sig_activation(self, z):
            """
            Compute the softmax of vector z, or row-wise for a matrix z.
            For numerical stability, subtract the maximum logit value from each
            row prior to exponentiation (see above).

            Parameters:
                `z` - a numpy array of shape (K,) or (N, K)

            Returns: a numpy array with the same shape as `z`, with the softmax
                activation applied to each row of `z`
            """

            if z.shape[1] == 1:
                ez = np.exp(z - np.max(z))
                return ez / ez.sum()
            else:
                ez = np.exp(z-np.max(z))
                sum = np.sum(ez, axis=1).reshape((z.shape[0], 1))
                return np.divide(ez, sum)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """

        act = None
        if self.activation == "logistic":
            act = self.sig_activation

        assert len(self.layer_matrices) >= 2

        # First Layer
        value = act(self.layer_matrices[0] @ X)

        # Deep Layers
        for i in range(self.num_layers - 2):
            value = act(self.layer_matrices[i+1] @ value)
        
        # Last Layer
        value = act(self.layer_matrices[-1] @ value)

        return value


def do_some_tests(m):
    # Get train, test data from get_data() in "./dataParsing.py"
    x_train, y_train, x_test, y_test = get_data()

    # Labels in Strings
    label_key = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]

    
    num_correct = 0
    # Change total to the number of test pieces you want to predict
    total = len(x_train)
    
    # Loop over nu
    for i in range(total):
        data_num = i
        features = np.array(x_train[data_num], dtype=float).reshape(-1, 1)
        label = label_key[np.argmax(y_train[data_num])]

        prediction = m.forward(features)
        predicted_label = label_key[np.argmax(prediction)]

        if label == predicted_label:
            num_correct += 1
            print("SUCCESFUL LABEL!!!: ",end="" )
        else:
            print("UNSUCCESFUL LABEL: ", end="")
        print(label, " - ", predicted_label)  
    print(num_correct / total, "%")


def get_model():
    lyr = (300, 300, 300)
    with open("./Weights/Hyperparameters.txt", "r") as f:
        lyr = eval(f.readline().strip().split(": ")[1])
    
    m = MLPModel(num_features = 138, num_hidden = lyr, num_classes = 4)
    return m


# GRADED IMPORTANT FUNCTION
def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    x = np.array(x, dtype=float).reshape(-1, 1)
    m = get_model()
    prediction = m.forward(x)
    label_key = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]
    predicted_label = label_key[np.argmax(prediction)]

    # return the prediction
    return predicted_label


# GRADED IMPORTANT FUNCTION
def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    m = get_model()
    label_key = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]

    data = csv.DictReader(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = m.forward(x)
        predicted_label = label_key[np.argmax(pred)]
        predictions.append(predicted_label)

    return predictions

#x_train, y_train, x_test, y_test = get_data()

#print(predict(x_train[0]))

# m = get_model()
# do_some_tests(m)

