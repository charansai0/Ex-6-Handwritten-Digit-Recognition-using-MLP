# Ex-6-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
/*Describe here*/


## Algorithm :


## Program:
Developed by :  v.charan sai

Reg no:212221240061
~~~
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
    def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
      test_prediction(0, W1, b1, W2, b2)print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    def make_predictions(X, W1, b1, W2, b2):
    A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    test_prediction(0, W1, b1, W2, b2)
~~~

## Output :
![203915185-e80620dc-00e5-4936-afe6-5925f1ac306a](https://user-images.githubusercontent.com/94296221/204022112-b9f912ce-fdac-4f1a-b445-4bea64b453df.png)
![203915222-a75f7f18-56c8-4b72-bb4f-9dc864cd7a1f](https://user-images.githubusercontent.com/94296221/204022117-acdd3470-98c5-4c47-9c60-8489df6b3a5d.png)
![203915235-bd54fc44-9d60-4287-80a5-ea1f067ce98b](https://user-images.githubusercontent.com/94296221/204022125-17271988-0266-4da3-b24a-a001559a2352.png)
![203915252-6bff0eb3-1491-492c-b109-03c6bbbe6e6a](https://user-images.githubusercontent.com/94296221/204022139-259883f2-7699-45ec-943f-7b06efd9b120.png)
![203915273-240d5c93-48aa-4473-ac99-029c7d9e92ba](https://user-images.githubusercontent.com/94296221/204022144-0c33fe1c-7704-4763-8a10-9502691f40af.png)
![203915290-d02c00a8-856c-49e7-bd43-bf639dccf42f](https://user-images.githubusercontent.com/94296221/204022150-63ee64c4-0ce7-468f-b45c-ad0954f15c59.png)


## Result:
Thus The Implementation of Handwritten Digit Recognition using MLP Is Executed Successfully.
