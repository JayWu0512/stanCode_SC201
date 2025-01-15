"""
File: titanic_deep_nn.py
Name: Jay
-----------------------------
This file demonstrates how to create a deep
neural network (5 layers NN) to train our
titanic data. Your code should use all the
constants and global variables.
You should see the following Acc if you
correctly implement the deep neural network
Acc: 0.8431372549019608 or 0.8235294117647058
-----------------------------
X.shape = (N0, m)
Y.shape = (1, m)
W1.shape -> (N0, N1)
W2.shape -> (N1, N2)
W3.shape -> (N2, N3)
W4.shape -> (N3, N4)
W5.shape -> (N4, N5)
B1.shape -> (N1, 1)
B2.shape -> (N2, 1)
B3.shape -> (N3, 1)
B4.shape -> (N4, 1)
B5.shape -> (N5, 1)
"""

from collections import defaultdict
import numpy as np

# Constants
TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 40000                   # This constant controls the total number of epochs
ALPHA = 0.01                         # This constant controls the learning rate α
L = 5                                # This number controls the number of layers in NN
NODES = {                            # This Dict[str: int] controls the number of nodes in each layer
    'N0': 6,
    'N1': 5,
    'N2': 4,
    'N3': 3,
    'N4': 2,
    'N5': 1
}


def main():
    """
    The final accuracy is 0.8431372549019608.
    """
    X_train, Y = data_preprocessing()

    # check the shape
    _, m = X_train.shape
    print('Y.shape', Y.shape)
    print('X.shape', X_train.shape)

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = normalize(X_train)

    # set X as A0 and get weights and biases that are trained in neural_network
    A = X
    weights, biases = neural_network(X, Y)

    # The last forward prop
    for i in range(1, L+1):
        K = weights["W"+str(i)].T.dot(A) + biases["B"+str(i)]
        # the last NODES(N5) should not use ReLU, so use if and else to control the situation
        if i < L:
            A = np.maximum(0, K)
        # the last NODE(N5) should be used as prediction
        else:
            scores = weights["W" + str(L)].T.dot(A) + biases["B" + str(L)]
            predictions = np.where(scores > 0, 1, 0)
            acc = np.equal(predictions, Y)
            print("Accuracy: ", np.sum(acc) / m)


def normalize(X):
    """
    :param X: numpy_array, the dimension is (num_phi, m)
    :return: numpy_array, the values are normalized, where the dimension is still (num_phi, m)
    """
    min_array = np.min(X, axis=1, keepdims=True)
    max_array = np.max(X, axis=1, keepdims=True)
    return (X - min_array) / (max_array - min_array)


def neural_network(X, Y):
    """
    :param X: numpy_array, the array holding all the training data
    :param Y: numpy_array, the array holding all the ture labels in X
    :return (weights, bias): the tuple of parameters of this deep NN
             weights: Dict[str, float], key is 'W1', 'W2', ...
                                        value is the corresponding float
             bias: Dict[str, float], key is 'B1', 'B2', ...
                                     value is the corresponding float
    """
    n, m = X.shape
    np.random.seed(1)
    weights = {}
    biases = {}
    k = {}
    # set X as a0's default
    a = {"A0": X}
    d = {}

    # Initialize all the weights and biases
    for i in range(1, L+1):
        weights["W"+str(i)] = np.random.rand(NODES['N'+str(i-1)], NODES['N'+str(i)]) - 0.5
        biases["B"+str(i)] = np.random.rand(NODES['N'+str(i)], 1) - 0.5

    for epoch in range(NUM_EPOCHS):
        # Forward Pass
        for i in range(1, L):
            k["K"+str(i)] = np.dot(weights["W"+str(i)].T, a["A"+str(i-1)]) + biases["B"+str(i)]
            a["A"+str(i)] = np.maximum(0, k["K"+str(i)])
        scores = weights["W"+str(L)].T.dot(a["A"+str(L-1)])+biases["B"+str(L)]
        H = 1/(1+np.exp(-scores))

        # Backward Pass
        d["K"+str(L)] = (1/m) * np.sum(H-Y, axis=0, keepdims=True)
        d["W"+str(L)] = a["A"+str(L-1)].dot(d["K"+str(L)].T)
        d["B"+str(L)] = np.sum(d["K"+str(L)], axis=1, keepdims=True)
        for i in range(L-1, 0, -1):
            d["A"+str(i)] = np.dot(weights["W"+str(i+1)], d["K"+str(i+1)])
            d["K"+str(i)] = d["A"+str(i)] * np.where(k["K"+str(i)] > 0, 1, 0)
            d["W"+str(i)] = np.dot(a["A"+str(i-1)], d["K"+str(i)].T)
            d["B"+str(i)] = np.sum(d["K"+str(i)], axis=1, keepdims=True)

        # Updates all the weights and biases
        for i in range(1, L+1):
            weights["W"+str(i)] -= ALPHA * d["W"+str(i)]
            biases["B"+str(i)] -= ALPHA * d["B"+str(i)]

    return weights, biases


def data_preprocessing(mode='train'):
    """
    :param mode: str, indicating if it's training mode or testing mode
    :return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
    """
    data_lst = []
    label_lst = []
    first_data = True
    if mode == 'train':
        with open(TRAIN, 'r') as f:
            for line in f:
                data = line.split(',')
                # ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
                if first_data:
                    first_data = False
                    continue
                if not data[6]:
                    continue
                label = [int(data[1])]
                if data[5] == 'male':
                    sex = 1
                else:
                    sex = 0
                # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
                passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
                data_lst.append(passenger_lst)
                label_lst.append(label)
    else:
        pass
    return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
    main()
