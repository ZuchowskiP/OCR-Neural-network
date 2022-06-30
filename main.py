import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.ttk import Label
from PIL import Image

#iniitial data load
data = pd.read_csv('./datasets/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)


data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

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

def gradient_descent(X, Y, alpha, iterations, Init):
    if Init == True:
        W1, b1, W2, b2 = init_params()
    else:
        W1, b1, W2, b2 = getValues()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, get_accuracy(predictions, Y)

# W1, b1, W2, b2, pred = gradient_descent(X_train, Y_train, 0.10, 500, True)
# # saving values for later use
# np.save('W1', W1)
# np.save('b1', b1)
# np.save('W2', W2)
# np.save('b2', b2)
# np.save('X_train', X_train)
# np.save('Y_train', Y_train)
# print(X_train.shape)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    # current_image = index
    prediction = make_predictions(index, W1, b1, W2, b2)

    # current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()
    return prediction

#####front
root = tk.Tk()
root.title('OCR \'Reczny\'')
root.geometry('500x300+50+50')


def chooseImage():
    filename = fd.askopenfilename(
        title='Select image',
        initialdir='/',
        filetypes=[("image", ".jpg"), ("image", '.png')]
    )
    return filename


def getLabel():
    content = inputtxt.get('1.0', 'end')
    return content


def getValues():
    W1 = np.load('./W1.npy')
    W2 = np.load('./W2.npy')
    b1 = np.load('./b1.npy')
    b2 = np.load('./b2.npy')
    return W1, b1, W2, b2


def saveValues(W1, b1, W2, b2):
    np.save('W1', W1)
    np.save('b1', b1)
    np.save('W2', W2)
    np.save('b2', b2)


def saveTrainValues(X_train, Y_train):
    np.save('X_train', X_train)
    np.save('Y_train', Y_train)


def getTrainValues():
    X_train = np.load('./X_train.npy')
    Y_train = np.load('./Y_train.npy')
    return X_train, Y_train

def proccessImage():
    img = chooseImage()
    img = np.asarray(Image.open(img).convert('L'))
    img = img.reshape(784, 1) / 255
    img = np.array(img)
    label = getLabel()
    W1, b1, W2, b2 = getValues()
    res = test_prediction(img, W1, b1, W2, b2)
    labelThird.config(text = res)
    if len(inputtxt.get("1.0", "end-1c")) != 0:
        label = int(label)
        X_train, Y_train = getTrainValues()
        for i in range(50):
            Y_train = np.append(Y_train, label)
            X_train = np.append(X_train, img, axis=1)
        saveTrainValues(X_train, Y_train)
        W1, b1, W2, b2, pred = gradient_descent(X_train, Y_train, 0.10, 50, False)
        labelSecond.config(text = 'Aktualna precyzja: ' + str(pred))
        saveValues(W1, b1, W2, b2)
    inputtxt.delete('1.0', 'end')

labelFirst = Label(root, text='Jesli chcesz aby model byl uczony wpisz rzeczywista wartosc przed wybraniem obrazka')
labelFirst.pack(ipadx=5, ipady=10)

inputtxt = tk.Text(root,
                   height=1,
                   width=20)

inputtxt.pack()

labelSecond = Label(root, text='')
labelSecond.pack(ipady=5, ipadx=5)

labelThird = Label(root, text='')
labelThird.pack(ipadx=5, ipady=5)

choose_button = ttk.Button(
    root,
    text='Choose file',
    command=proccessImage
)

choose_button.pack(
    ipadx=2,
    ipady=5,
    expand=True
)


root.mainloop()
