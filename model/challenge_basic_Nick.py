from FinalPred.dataParsing import get_data

from sklearn.neural_network import MLPClassifier
import glob
import os
import numpy as np


x_train, y_train, x_test, y_test = get_data()





# Train and evaluate classifiers
# -----------------------------------------------------------------------

# Hyper parameters
lyr = (100,)
act = "logistic"
alpha = 0.1
clf = MLPClassifier(max_iter=100, hidden_layer_sizes=lyr, activation=act, verbose=True, alpha=alpha )

# Train and test
clf.fit(x_train, y_train)
train_acc = clf.score(x_train, y_train)
test_acc = clf.score(x_test, y_test)

# All weights
#import pdb; pdb.set_trace()
coef = clf.coefs_
biases = clf.intercepts_


# Delete all files in ./FinalPred/Weights
for f in glob.glob('./FinalPred/Weights/*'):
    os.remove(f)
for i, layer in enumerate(coef):
    with open(f"./FinalPred/Weights/Layer{i}weights.txt", "w+") as f:
        for j, node_weights in enumerate(layer):
            f.write(f"Node {j}: \n")
            f.write(str(node_weights) +"\n")

for i, bias_layer in enumerate(biases):
    with open(f"./FinalPred/Weights/Layer{i}biases.txt", "w+") as f:
        f.write(str(bias_layer.tolist()) + "\n")


# ----- Testing single inputs
label_key = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]
data_num = 0
features = np.array([x_train[data_num]], dtype=float)
label = label_key[np.argmax(y_train[data_num])]
prediction = label_key[np.argmax(clf.predict(features))]
print(f"Prediction - Label  | {prediction} - {label}")
# ----- Testing single inputs

# Write hyperparameters to "Hyperparameters.txt"
with open(f"./FinalPred/Weights/Hyperparameters.txt", "w+") as f:
    f.write(f"Layers: {lyr}\n")
    f.write(f"Length of featuers: {len(features[0])}\n")
    f.write(f"Activation: {act}\n")
    f.write(f"Alpha: {alpha}\n")
    f.write(f"{type(clf).__name__} train acc: {train_acc}\n")
    f.write(f"{type(clf).__name__} test acc: {test_acc}\n")


# Print output
print("Layers: ", lyr)
print("activation: ", act)
print("alpha: ", alpha)
print("Length of featuers: ", len(features[0]))
print(f"{type(clf).__name__} train acc: {train_acc}")
print(f"{type(clf).__name__} test acc: {test_acc}")


