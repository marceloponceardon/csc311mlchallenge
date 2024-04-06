import os
import numpy as np
from dataParsing import get_data






class MLPModel(object):
    def __init__(self, num_features=128*20, num_hidden=(300, 300, 300, 300), num_classes=128):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = num_features
        self.num_hidden_1 = num_hidden[0]
        self.num_hidden_2 = num_hidden[1]
        self.num_hidden_3 = num_hidden[2]
        self.num_hidden_4 = num_hidden[3]

        self.num_classes = num_classes
        self.num_layers = len(num_hidden) + 1

     
        self.W1 = np.zeros([self.num_hidden_1, self.num_features])
        self.W2 = np.zeros([self.num_hidden_2, self.num_hidden_1])
        self.W3 = np.zeros([self.num_hidden_3, self.num_hidden_2])
        self.W4 = np.zeros([self.num_hidden_4, self.num_hidden_3])
        self.W5 = np.zeros([self.num_classes, self.num_hidden_4])
        self.weight_matrices = [self.W1, self.W2, self.W3, self.W4, self.W5]

        # Read weights from files and set it into matrices
        for i in range(self.num_layers):
            weight_matrix = self.weight_matrices[i]
            with open(f"./Weights/Layer{i}weights.txt", "r") as f:
                weight_col_num = 0
                col_weights = []
                for line in f.readlines():
                    if "Node " in line:
                        if col_weights:
                            #print(col_weights)
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
                                   


    def activation(self, z):
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
        #import pdb; pdb.set_trace()
        L1 = self.activation(self.W1 @ X)
        L2 = self.activation(self.W2 @ L1)
        L3 = self.activation(self.W3 @ L2)
        L4 = self.activation(self.W4 @ L3)
        L5 = self.activation(self.W5 @ L4)

        return L5

    
    def loss(self, ts):
        """
        Compute the average cross-entropy loss, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]




def do_forward_pass(model, X):
    """
    Compute the forward pass to produce prediction for inputs.

    This function also keeps some of the intermediate values in
    the neural network computation, to make computing gradients easier.

    For the ReLU activation, you may find the function `np.maximum` helpful

    Parameters:
        `model` - An instance of the class MLPModel
        `X` - A numpy array of shape (model.num_features, 1)

    Returns: A numpy array of predictions of shape (N, model.num_classes)
    """
    

    # model.N = X.shape[0]
    # model.X = X
    # #print("w1 transpse: ", model.W1.transpose().shape, "x:", model.X.shape)
    # if len(model.b1.shape) == 1:
    #     b1 = model.b1.reshape(-1, 1)
    # else:
    #     b1 = model.b1
    # if len(model.b2.shape) == 1:
    #     b2 = model.b2.reshape(-1, 1)
    # else:
    #     b2 = model.b2.transpose()

    # # print(model.b1.shape)
    # model.m = (model.W1.transpose() @ model.X.transpose()) + b1  # TODO - the hidden state value (pre-activation)
    # model.h = np.maximum(model.m, np.zeros(model.m.shape)) # TODO - the hidden state value (post ReLU activation)
    # model.z = (model.W2.transpose() @ model.h) + b2       # TODO - the logit scores (pre-activation)
    # model.y = softmax(model.z).transpose()                                        # TODO - the class probabilities (post-activation)

    # #print("w1.transpose: ", model.W1.transpose(), "b1: ", model.b1, "x: ", model.X, "N: ", model.N, "m: ", model.m, "h: ", model.h, "z: ", model.z, "y: ", model.y, "b1: ", sep="\n")
    # #print(model.y.shape, model.N)
    # return model.y


m = MLPModel(num_features = 138, num_hidden = (300, 300, 300, 300), num_classes = 4)
x_train, y_train, x_test, y_test = get_data()

label_key = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]
data_num = 50

print(len(y_train))
num_correct = 0
total = len(x_train)
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


# features = np.array(x_train[data_num], dtype=float).reshape(-1, 1)
# label = label_key[np.argmax(y_train[50])]

# prediction = m.forward(features)
# predicted_label = np.argmax(prediction)

# if label == predicted_label:
#     num_correct += 1
#     print("SUCCESFUL LABEL!!!: ",end="" )
# else:
#     print("UNSUCCESFUL LABEL: ", end="")
# print(label, " - ", predicted_label)  