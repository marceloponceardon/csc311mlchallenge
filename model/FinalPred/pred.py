import os
import numpy as np
from dataParsing import get_data




class MLPModel(object):
    def __init__(self, num_features=138, num_hidden=(300, 300, 300, 300), num_classes=4):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = num_features
        self.layer_format = num_hidden
        self.num_classes = num_classes
        self.num_layers = len(num_hidden) + 1


        self.layer_matrices = []

        # Featuers to first hidden nodes layer
        self.layer_matrices.append(np.zeros((self.layer_format[0], self.num_features)))
        for i in range(self.num_layers - 2):
            self.layer_matrices.append(np.zeros((self.layer_format[i+1], self.layer_format[i])))

        # Last hidden layer to classes layer
        self.layer_matrices.append(np.zeros((self.num_classes, self.layer_format[-1])))



        # Read weights from files and set it into matrices
        for i in range(self.num_layers):
            weight_matrix = self.layer_matrices[i]
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
            print(weight_matrix)
            break                   


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
        # Update
        return 
        #import pdb; pdb.set_trace()
        L1 = self.sig_activation(self.W1 @ X)
        L2 = self.sig_activation(self.W2 @ L1)
        L3 = self.sig_activation(self.W3 @ L2)
        L4 = self.sig_activation(self.W4 @ L3)
        L5 = self.sig_activation(self.W5 @ L4)

        return L5





#m = MLPModel(num_features = 138, num_hidden = (300, 300, 300, 300), num_classes = 4)

m = MLPModel(num_features = 138, num_hidden = (5,), num_classes = 4)

raise Exception
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


features = np.array(x_train[data_num], dtype=float).reshape(-1, 1)
label = label_key[np.argmax(y_train[50])]

prediction = m.forward(features)
predicted_label = np.argmax(prediction)

if label == predicted_label:
    num_correct += 1
    print("SUCCESFUL LABEL!!!: ",end="" )
else:
    print("UNSUCCESFUL LABEL: ", end="")
print(label, " - ", predicted_label)  